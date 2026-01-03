using Microsoft.ML.OnnxRuntime;

using Newtonsoft.Json;

using System.Collections.Concurrent;
using System.Data;
using System.Data.Common;
using Microsoft.Data.Sqlite;
using System.Diagnostics;
using System.Text;

namespace LocalRAG
{
    public class EmbeddingDatabaseNew : IDisposable
    {
        private readonly RAGConfiguration _config;
        private readonly EmbedderClassNew _embedder;
        private readonly ConcurrentDictionary<string, CacheEntry<object>> _cache;
        private readonly ConcurrentQueue<(FeedbackDatabaseValues data, TaskCompletionSource<bool> completion)> _updateQueue;
        private readonly SemaphoreSlim _queueSemaphore;
        private readonly SemaphoreSlim _processingSemaphore;
        private readonly CancellationTokenSource _cancellationTokenSource;
        private readonly System.Threading.Timer _cleanupTimer;
        private readonly StringBuilder _logBuilder;

        private List<List<float[]>> _hashFunctions;
        private List<Dictionary<int, List<int>>> _hashTables;
        private Task _processingTask;
        private bool _isDisposing;

        // BERT embedding dimension: 768 for base, 1024 for large
        // This should match the actual model output dimension
        private const int EmbeddingDimension = 768;

        // Column ordinals for reader optimization
        private int _idIndex;
        private int _requestIdIndex;
        private int _userMessageTypeIndex;
        private int _assistantMessageTypeIndex;
        private int _ratingIndex;
        private int _metaDataIndex;
        private int _requestIndex;
        private int _textResponseIndex;
        private int _toolUseTextResponseIndex;
        private int _toolNameIndex;
        private int _toolContentIndex;
        private int _toolResultIndex;
        private int _codeIndex;
        private int _summaryIndex;
        private int _commentIndex;
        private int _errorsIndex;
        private int _requestStatusIndex;
        private int _requestEmbeddingIndex;
        private int _requestEmbeddingListIndex;
        private int _textResponseEmbeddingIndex;
        private int _textResponseEmbeddingListIndex;
        private int _toolUseTextResponseEmbeddingIndex;
        private int _toolUseTextResponseEmbeddingListIndex;
        private int _summaryEmbeddingIndex;
        private int _summaryEmbeddingListIndex;
        private int _timestampIndex;
        private bool _ordinalsInitialized = false;

        private static readonly HashSet<string> StopWords = new HashSet<string>
        {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "it", "as", "that", "this", "these", "those", "are", "was", "were", "be",
            "from", "up", "about", "into", "over", "after"
        };

        public EmbeddingDatabaseNew(RAGConfiguration config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _embedder = new EmbedderClassNew(config);
            _logBuilder = new StringBuilder();
            _cache = new ConcurrentDictionary<string, CacheEntry<object>>();
            _updateQueue = new ConcurrentQueue<(FeedbackDatabaseValues, TaskCompletionSource<bool>)>();
            _queueSemaphore = new SemaphoreSlim(_config.MaxQueueSize, _config.MaxQueueSize);
            _processingSemaphore = new SemaphoreSlim(1, 1);
            _cancellationTokenSource = new CancellationTokenSource();
            _isDisposing = false;

            _cleanupTimer = new System.Threading.Timer(_ => CleanupCache(), null, TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(5));

            InitializeLSH();
            _ = InitializeDatabaseAsync();
            StartProcessor();
        }

        #region Initialization

        private async Task InitializeDatabaseAsync()
        {
            await CreateTableAsync();
            await InitializeOrdinalsAsync();
        }

        private async Task CreateTableAsync()
        {
            // Using regular table instead of FTS5 for better compatibility with embeddings
            // FTS5 is added as a separate virtual table for text search
            const string createMainTable = @"
CREATE TABLE IF NOT EXISTS embeddings (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    RequestID TEXT UNIQUE,
    UserMessageType TEXT,
    AssistantMessageType TEXT,
    Rating INTEGER DEFAULT 0,
    MetaData TEXT,
    Request TEXT,
    TextResponse TEXT,
    ToolUseTextResponse TEXT,
    ToolName TEXT,
    ToolContent TEXT,
    ToolResult TEXT,
    Code TEXT,
    Summary TEXT,
    Comment TEXT,
    Errors TEXT,
    RequestStatus TEXT,
    RequestEmbedding TEXT,
    TextResponseEmbedding TEXT,
    ToolUseTextResponseEmbedding TEXT,
    SummaryEmbedding TEXT,
    RequestEmbeddingList TEXT,
    TextResponseEmbeddingList TEXT,
    ToolUseTextResponseEmbeddingList TEXT,
    SummaryEmbeddingList TEXT,
    Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_embeddings_requestid ON embeddings(RequestID);
CREATE INDEX IF NOT EXISTS idx_embeddings_timestamp ON embeddings(Timestamp);
";

            // FTS5 virtual table for full-text search (shadows main table)
            const string createFtsTable = @"
CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_fts USING fts5(
    RequestID,
    Request,
    TextResponse,
    ToolUseTextResponse,
    Summary,
    content='embeddings',
    content_rowid='Id'
);
";

            // Triggers to keep FTS in sync
            const string createTriggers = @"
CREATE TRIGGER IF NOT EXISTS embeddings_ai AFTER INSERT ON embeddings BEGIN
    INSERT INTO embeddings_fts(rowid, RequestID, Request, TextResponse, ToolUseTextResponse, Summary)
    VALUES (new.Id, new.RequestID, new.Request, new.TextResponse, new.ToolUseTextResponse, new.Summary);
END;

CREATE TRIGGER IF NOT EXISTS embeddings_ad AFTER DELETE ON embeddings BEGIN
    INSERT INTO embeddings_fts(embeddings_fts, rowid, RequestID, Request, TextResponse, ToolUseTextResponse, Summary)
    VALUES ('delete', old.Id, old.RequestID, old.Request, old.TextResponse, old.ToolUseTextResponse, old.Summary);
END;

CREATE TRIGGER IF NOT EXISTS embeddings_au AFTER UPDATE ON embeddings BEGIN
    INSERT INTO embeddings_fts(embeddings_fts, rowid, RequestID, Request, TextResponse, ToolUseTextResponse, Summary)
    VALUES ('delete', old.Id, old.RequestID, old.Request, old.TextResponse, old.ToolUseTextResponse, old.Summary);
    INSERT INTO embeddings_fts(rowid, RequestID, Request, TextResponse, ToolUseTextResponse, Summary)
    VALUES (new.Id, new.RequestID, new.Request, new.TextResponse, new.ToolUseTextResponse, new.Summary);
END;
";

            using var connection = await GetConnectionAsync();
            using var transaction = connection.BeginTransaction();

            try
            {
                using (var cmd = new SqliteCommand(createMainTable, connection, transaction))
                    await cmd.ExecuteNonQueryAsync();

                using (var cmd = new SqliteCommand(createFtsTable, connection, transaction))
                    await cmd.ExecuteNonQueryAsync();

                using (var cmd = new SqliteCommand(createTriggers, connection, transaction))
                    await cmd.ExecuteNonQueryAsync();

                transaction.Commit();
            }
            catch (Exception ex)
            {
                transaction.Rollback();
                LogMessage($"Error creating tables: {ex.Message}");
                throw;
            }
        }

        private void InitializeLSH()
        {
            _hashFunctions = new List<List<float[]>>();
            _hashTables = new List<Dictionary<int, List<int>>>();
            var random = new Random(42); // Fixed seed for reproducibility

            for (int i = 0; i < _config.NumberOfHashTables; i++)
            {
                var tableFunctions = new List<float[]>();
                for (int j = 0; j < _config.NumberOfHashFunctions; j++)
                {
                    // IMPORTANT: Use EmbeddingDimension (768/1024), NOT MaxSequenceLength (512)
                    // MaxSequenceLength is the token count, EmbeddingDimension is the vector size
                    var vector = new float[EmbeddingDimension];
                    for (int k = 0; k < EmbeddingDimension; k++)
                    {
                        // Box-Muller transform for Gaussian random numbers
                        double u1 = 1.0 - random.NextDouble();
                        double u2 = 1.0 - random.NextDouble();
                        vector[k] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
                    }
                    tableFunctions.Add(vector);
                }
                _hashFunctions.Add(tableFunctions);
                _hashTables.Add(new Dictionary<int, List<int>>());
            }
        }

        private async Task InitializeOrdinalsAsync()
        {
            const string query = "SELECT * FROM embeddings LIMIT 0";

            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand(query, connection);
            using var reader = await command.ExecuteReaderAsync();

            var ordinalSetters = new Dictionary<string, Action<int>>
            {
                {"Id", index => _idIndex = index},
                {"RequestID", index => _requestIdIndex = index},
                {"UserMessageType", index => _userMessageTypeIndex = index},
                {"AssistantMessageType", index => _assistantMessageTypeIndex = index},
                {"Rating", index => _ratingIndex = index},
                {"MetaData", index => _metaDataIndex = index},
                {"Request", index => _requestIndex = index},
                {"TextResponse", index => _textResponseIndex = index},
                {"ToolUseTextResponse", index => _toolUseTextResponseIndex = index},
                {"ToolName", index => _toolNameIndex = index},
                {"ToolContent", index => _toolContentIndex = index},
                {"ToolResult", index => _toolResultIndex = index},
                {"Code", index => _codeIndex = index},
                {"Summary", index => _summaryIndex = index},
                {"Comment", index => _commentIndex = index},
                {"Errors", index => _errorsIndex = index},
                {"RequestStatus", index => _requestStatusIndex = index},
                {"RequestEmbedding", index => _requestEmbeddingIndex = index},
                {"RequestEmbeddingList", index => _requestEmbeddingListIndex = index},
                {"TextResponseEmbedding", index => _textResponseEmbeddingIndex = index},
                {"TextResponseEmbeddingList", index => _textResponseEmbeddingListIndex = index},
                {"ToolUseTextResponseEmbedding", index => _toolUseTextResponseEmbeddingIndex = index},
                {"ToolUseTextResponseEmbeddingList", index => _toolUseTextResponseEmbeddingListIndex = index},
                {"SummaryEmbedding", index => _summaryEmbeddingIndex = index},
                {"SummaryEmbeddingList", index => _summaryEmbeddingListIndex = index},
                {"Timestamp", index => _timestampIndex = index}
            };

            foreach (var field in ordinalSetters)
            {
                try
                {
                    field.Value(reader.GetOrdinal(field.Key));
                }
                catch (IndexOutOfRangeException)
                {
                    LogMessage($"Column {field.Key} not found in database");
                }
            }

            _ordinalsInitialized = true;
        }

        #endregion

        #region Connection Management

        private string GetConnectionString() =>
            $"Data Source={_config.DatabasePath};Pooling=True;Cache=Shared";

        private async Task<SqliteConnection> GetConnectionAsync()
        {
            var connection = new SqliteConnection(GetConnectionString());
            await connection.OpenAsync();
            return connection;
        }

        #endregion

        #region Public API - Add/Update

        public async Task AddRequestToEmbeddingDatabaseAsync(string? requestId, string theRequest, bool embed = false)
        {
            var data = new FeedbackDatabaseValues
            {
                RequestID = requestId,
                UserMessageType = "text",
                Request = theRequest,
                Embed = embed,
                EmbedRequest = embed,
                EmbedRequestList = embed
            };
            await AddDataToEmbeddingDatabaseAsync(data);
        }

        public async Task AddRequestToolResultToEmbeddingDatabaseAsync(string? requestId, string theRequest, bool embed = false)
        {
            var data = new FeedbackDatabaseValues
            {
                RequestID = requestId,
                UserMessageType = "tool_result",
                Request = theRequest,
                Embed = embed,
                EmbedRequest = embed,
                EmbedRequestList = embed
            };
            await AddDataToEmbeddingDatabaseAsync(data);
        }

        public async Task UpdateTextResponse(string requestId, string message, bool embed = false)
        {
            var data = new FeedbackDatabaseValues
            {
                RequestID = requestId,
                TextResponse = message,
                Embed = embed,
                EmbedTextResponse = embed,
                EmbedTextResponseList = embed
            };
            await UpdateDataInEmbeddingDatabaseAsync(data);
        }

        public async Task UpdateToolUseTextResponse(string requestId, string message, bool embed = false)
        {
            var data = new FeedbackDatabaseValues
            {
                RequestID = requestId,
                ToolUseTextResponse = message,
                UserMessageType = "tool_use",
                Embed = embed,
                EmbedToolUseTextResponse = embed,
                EmbedToolUseTextResponseList = embed
            };
            await UpdateDataInEmbeddingDatabaseAsync(data);
        }

        public async Task UpdateToolContent(string requestId, string toolContent)
        {
            var data = new FeedbackDatabaseValues
            {
                RequestID = requestId,
                ToolContent = toolContent
            };
            await UpdateDataInEmbeddingDatabaseAsync(data);
        }

        public async Task UpdateToolResult(string requestId, string message, bool embed = false)
        {
            var data = new FeedbackDatabaseValues
            {
                RequestID = requestId,
                ToolResult = message,
                Embed = embed
            };
            await UpdateDataInEmbeddingDatabaseAsync(data);
        }

        public async Task UpdateSummary(string requestId, string message, bool embed = false)
        {
            var data = new FeedbackDatabaseValues
            {
                RequestID = requestId,
                Summary = message,
                Embed = embed,
                EmbedSummary = embed,
                EmbedSummaryList = embed
            };
            await UpdateDataInEmbeddingDatabaseAsync(data);
        }

        public async Task UpdateDataInEmbeddingDatabaseAsync(FeedbackDatabaseValues data)
        {
            if (string.IsNullOrEmpty(data.RequestID))
                throw new ArgumentException("RequestID is required.", nameof(data.RequestID));

            InvalidateCache(data.RequestID);

            var completionSource = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            await _queueSemaphore.WaitAsync();

            try
            {
                _updateQueue.Enqueue((data, completionSource));
            }
            finally
            {
                _queueSemaphore.Release();
            }

            await completionSource.Task;
        }

        #endregion

        #region Public API - Search

        public async Task<List<FeedbackDatabaseValues>> SearchEmbeddingsAsync(
            string searchText,
            int topK,
            float minimumSimilarity = 0.5f,
            int searchLevel = 2)
        {
            if (string.IsNullOrWhiteSpace(searchText))
                return new List<FeedbackDatabaseValues>();

            // Get results from multiple search methods
            var ftsResults = await SearchWithFTSAsync(searchText, topK, searchLevel);
            var lshResults = await SearchWithLSHAsync(searchText, topK, minimumSimilarity);
            var memoryResults = await SearchMemoryBasedAsync(searchText, topK);

            // Combine and deduplicate results
            var combinedResults = new Dictionary<string, FeedbackDatabaseValues>();

            foreach (var result in ftsResults.Concat(lshResults).Concat(memoryResults))
            {
                if (result.RequestID == null) continue;

                if (!combinedResults.ContainsKey(result.RequestID))
                {
                    combinedResults[result.RequestID] = result;
                }
                else
                {
                    // Keep the higher similarity score
                    combinedResults[result.RequestID].Similarity =
                        Math.Max(combinedResults[result.RequestID].Similarity ?? 0, result.Similarity ?? 0);
                }
            }

            return combinedResults.Values
                .Where(x => x.Similarity >= minimumSimilarity)
                .OrderByDescending(x => x.Similarity)
                .Take(topK)
                .ToList();
        }

        public async Task<List<FeedbackDatabaseValues>> SearchMemoryBasedAsync(string searchText, int topK)
        {
            var similarResults = await _embedder.SearchSimilarAsync(searchText, topK);
            var feedbackResults = new List<FeedbackDatabaseValues>();

            foreach (var result in similarResults)
            {
                if (!result.Metadata.Tags.TryGetValue("text", out var originalText))
                    continue;

                var matchingEntries = await SearchByTextContentAsync(originalText);
                foreach (var entry in matchingEntries)
                {
                    entry.Similarity = result.Score;
                    feedbackResults.Add(entry);
                }
            }

            return feedbackResults
                .OrderByDescending(x => x.Similarity)
                .Take(topK)
                .ToList();
        }

        public async Task<FeedbackDatabaseValues?> GetFeedbackDataByRequestIdAsync(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return null;

            // Check cache first
            var cacheKey = $"feedback_{requestId}";
            if (_cache.TryGetValue(cacheKey, out var cached) && cached.Expiry > DateTime.UtcNow)
                return (FeedbackDatabaseValues?)cached.Data;

            const string query = "SELECT * FROM embeddings WHERE RequestID = @RequestID";

            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand(query, connection);
            command.Parameters.AddWithValue("@RequestID", requestId);

            using var reader = await command.ExecuteReaderAsync();
            if (await reader.ReadAsync())
            {
                var result = ReadFeedbackFromReader(reader);

                // Cache the result
                _cache[cacheKey] = new CacheEntry<object>
                {
                    Data = result,
                    Expiry = DateTime.UtcNow.AddMinutes(15)
                };

                return result;
            }

            return null;
        }

        public async Task<FeedbackDatabaseValues?> GetFeedbackDataByIdAsync(int id)
        {
            const string query = "SELECT * FROM embeddings WHERE Id = @Id";

            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand(query, connection);
            command.Parameters.AddWithValue("@Id", id);

            using var reader = await command.ExecuteReaderAsync();
            return await reader.ReadAsync() ? ReadFeedbackFromReader(reader) : null;
        }

        public async Task<List<(string request, string textResponse, string toolUseTextResponse, string toolContent, string toolResult, string requestID)>>
            GetConversationHistoryAsync(int messageCount = 100)
        {
            var results = new List<(string, string, string, string, string, string)>();

            const string query = @"
                SELECT Request, TextResponse, ToolUseTextResponse, ToolContent, ToolResult, RequestID 
                FROM embeddings 
                ORDER BY Id DESC 
                LIMIT @Count";

            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand(query, connection);
            command.Parameters.AddWithValue("@Count", messageCount);

            using var reader = await command.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                var request = reader.IsDBNull(0) ? "" : reader.GetString(0);
                var textResponse = reader.IsDBNull(1) ? "" : reader.GetString(1);
                var toolUseTextResponse = reader.IsDBNull(2) ? "" : reader.GetString(2);
                var toolContent = reader.IsDBNull(3) ? "" : reader.GetString(3);
                var toolResult = reader.IsDBNull(4) ? "" : reader.GetString(4);
                var requestId = reader.IsDBNull(5) ? "" : reader.GetString(5);

                if (!string.IsNullOrEmpty(textResponse) || !string.IsNullOrEmpty(toolUseTextResponse))
                {
                    results.Add((request, textResponse, toolUseTextResponse, toolContent, toolResult, requestId));
                }
            }

            results.Reverse();
            return results;
        }

        #endregion

        #region Public API - Embedding Maintenance

        /// <summary>
        /// Regenerates embeddings for records in the database.
        /// Use this after fixing embedding generation logic to update old/incorrect embeddings.
        /// </summary>
        /// <param name="mode">
        /// "missing" - Only generate embeddings for records that have null embeddings
        /// "all" - Regenerate all embeddings (replaces existing)
        /// "both" - Same as "all", regenerates everything
        /// </param>
        /// <param name="batchSize">Number of records to process before yielding progress</param>
        /// <param name="progress">Optional progress callback (processed, total, currentRequestId)</param>
        /// <param name="cancellationToken">Cancellation token to stop the operation</param>
        /// <returns>Summary of the backfill operation</returns>
        public async Task<BackfillResult> BackfillEmbeddingsAsync(
            string mode = "missing",
            int batchSize = 10,
            Action<int, int, string>? progress = null,
            CancellationToken cancellationToken = default)
        {
            var result = new BackfillResult();
            var stopwatch = Stopwatch.StartNew();

            bool replaceExisting = mode.ToLower() is "all" or "both" or "replace";

            LogMessage($"Starting embedding backfill - Mode: {mode}, ReplaceExisting: {replaceExisting}");

            // Get all records that need processing
            var recordsToProcess = await GetRecordsForBackfillAsync(replaceExisting);
            result.TotalRecords = recordsToProcess.Count;

            LogMessage($"Found {result.TotalRecords} records to process");

            foreach (var record in recordsToProcess)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    result.WasCancelled = true;
                    break;
                }

                try
                {
                    var updated = await RegenerateEmbeddingsForRecordAsync(record, replaceExisting);

                    if (updated)
                    {
                        result.SuccessCount++;
                        // Clear LSH tables for this record and re-add
                        await RebuildLSHForRecordAsync(record.Id);
                    }
                    else
                    {
                        result.SkippedCount++;
                    }
                }
                catch (Exception ex)
                {
                    result.FailedCount++;
                    result.Errors.Add($"RequestID {record.RequestID}: {ex.Message}");
                    LogMessage($"Error processing {record.RequestID}: {ex.Message}");
                }

                result.ProcessedCount++;
                progress?.Invoke(result.ProcessedCount, result.TotalRecords, record.RequestID ?? "unknown");

                // Yield occasionally to prevent blocking
                if (result.ProcessedCount % batchSize == 0)
                {
                    await Task.Delay(10, cancellationToken);
                }
            }

            stopwatch.Stop();
            result.ElapsedTime = stopwatch.Elapsed;

            LogMessage($"Backfill complete - Processed: {result.ProcessedCount}, Success: {result.SuccessCount}, " +
                      $"Failed: {result.FailedCount}, Skipped: {result.SkippedCount}, Time: {result.ElapsedTime}");

            return result;
        }

        /// <summary>
        /// Gets records that need embedding regeneration
        /// </summary>
        private async Task<List<FeedbackDatabaseValues>> GetRecordsForBackfillAsync(bool includeExisting)
        {
            var records = new List<FeedbackDatabaseValues>();

            string query = includeExisting
                ? "SELECT * FROM embeddings ORDER BY Id"
                : @"SELECT * FROM embeddings
                    WHERE RequestEmbedding IS NULL
                       OR TextResponseEmbedding IS NULL
                       OR (TextResponse IS NOT NULL AND TextResponseEmbedding IS NULL)
                       OR (ToolUseTextResponse IS NOT NULL AND ToolUseTextResponseEmbedding IS NULL)
                       OR (Summary IS NOT NULL AND SummaryEmbedding IS NULL)
                    ORDER BY Id";

            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand(query, connection);

            using var reader = await command.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                records.Add(ReadFeedbackFromReader(reader));
            }

            return records;
        }

        /// <summary>
        /// Regenerates embeddings for a single record
        /// </summary>
        private async Task<bool> RegenerateEmbeddingsForRecordAsync(FeedbackDatabaseValues record, bool replaceExisting)
        {
            bool anyUpdated = false;
            var updates = new Dictionary<string, string>();

            // Debug: Log what we're working with
            Debug.WriteLine($"Processing record {record.RequestID}:");
            Debug.WriteLine($"  - Request: {(string.IsNullOrEmpty(record.Request) ? "EMPTY" : $"{record.Request.Length} chars")}");
            Debug.WriteLine($"  - TextResponse: {(string.IsNullOrEmpty(record.TextResponse) ? "EMPTY" : $"{record.TextResponse.Length} chars")}");
            Debug.WriteLine($"  - RequestEmbedding: {(record.RequestEmbedding == null ? "NULL" : $"{record.RequestEmbedding.Length} dims")}");
            Debug.WriteLine($"  - ReplaceExisting: {replaceExisting}");

            // Regenerate Request embedding
            if (!string.IsNullOrEmpty(record.Request) &&
                (replaceExisting || record.RequestEmbedding == null))
            {
                Debug.WriteLine($"  -> Generating Request embedding...");
                var embedding = await _embedder.TryGetEmbeddingsAsync(record.Request);
                if (embedding != null)
                {
                    Debug.WriteLine($"  -> Got embedding with {embedding.Length} dimensions");
                    updates["RequestEmbedding"] = JsonConvert.SerializeObject(embedding);
                    anyUpdated = true;
                }
                else
                {
                    Debug.WriteLine($"  -> Embedding returned NULL");
                }

                var embeddingList = await TryGetEmbeddingListAsync(record.Request);
                if (embeddingList != null)
                {
                    updates["RequestEmbeddingList"] = JsonConvert.SerializeObject(embeddingList);
                }
            }
            else
            {
                Debug.WriteLine($"  -> SKIPPING Request (empty or already has embedding)");
            }

            // Regenerate TextResponse embedding
            if (!string.IsNullOrEmpty(record.TextResponse) &&
                (replaceExisting || record.TextResponseEmbedding == null))
            {
                var embedding = await _embedder.TryGetEmbeddingsAsync(record.TextResponse);
                if (embedding != null)
                {
                    updates["TextResponseEmbedding"] = JsonConvert.SerializeObject(embedding);
                    anyUpdated = true;
                }

                var embeddingList = await TryGetEmbeddingListAsync(record.TextResponse);
                if (embeddingList != null)
                {
                    updates["TextResponseEmbeddingList"] = JsonConvert.SerializeObject(embeddingList);
                }
            }

            // Regenerate ToolUseTextResponse embedding
            if (!string.IsNullOrEmpty(record.ToolUseTextResponse) &&
                (replaceExisting || record.ToolUseTextResponseEmbedding == null))
            {
                var embedding = await _embedder.TryGetEmbeddingsAsync(record.ToolUseTextResponse);
                if (embedding != null)
                {
                    updates["ToolUseTextResponseEmbedding"] = JsonConvert.SerializeObject(embedding);
                    anyUpdated = true;
                }

                var embeddingList = await TryGetEmbeddingListAsync(record.ToolUseTextResponse);
                if (embeddingList != null)
                {
                    updates["ToolUseTextResponseEmbeddingList"] = JsonConvert.SerializeObject(embeddingList);
                }
            }

            // Regenerate Summary embedding
            if (!string.IsNullOrEmpty(record.Summary) &&
                (replaceExisting || record.SummaryEmbedding == null))
            {
                var embedding = await _embedder.TryGetEmbeddingsAsync(record.Summary);
                if (embedding != null)
                {
                    updates["SummaryEmbedding"] = JsonConvert.SerializeObject(embedding);
                    anyUpdated = true;
                }

                var embeddingList = await TryGetEmbeddingListAsync(record.Summary);
                if (embeddingList != null)
                {
                    updates["SummaryEmbeddingList"] = JsonConvert.SerializeObject(embeddingList);
                }
            }

            // Apply updates if any
            if (updates.Count > 0)
            {
                Debug.WriteLine($"  -> Saving {updates.Count} embedding updates to database...");
                await UpdateEmbeddingColumnsAsync(record.RequestID!, updates);
                Debug.WriteLine($"  -> Database update complete");
            }
            else
            {
                Debug.WriteLine($"  -> No updates to save");
            }

            Debug.WriteLine($"  -> Record complete. anyUpdated={anyUpdated}");
            return anyUpdated;
        }

        /// <summary>
        /// Helper to get embedding list with error handling
        /// </summary>
        private async Task<List<float[]>?> TryGetEmbeddingListAsync(string text)
        {
            try
            {
                return await _embedder.SplitStringIntoEmbeddingsListAsync(text);
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Updates specific embedding columns for a record
        /// </summary>
        private async Task UpdateEmbeddingColumnsAsync(string requestId, Dictionary<string, string> updates)
        {
            if (updates.Count == 0) return;

            var setClauses = string.Join(", ", updates.Keys.Select(k => $"{k} = @{k}"));
            var query = $"UPDATE embeddings SET {setClauses} WHERE RequestID = @RequestID";

            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand(query, connection);

            command.Parameters.AddWithValue("@RequestID", requestId);
            foreach (var kvp in updates)
            {
                command.Parameters.AddWithValue($"@{kvp.Key}", kvp.Value);
            }

            await command.ExecuteNonQueryAsync();
            InvalidateCache(requestId);
        }

        /// <summary>
        /// Rebuilds LSH index entries for a specific record
        /// </summary>
        private async Task RebuildLSHForRecordAsync(int recordId)
        {
            var record = await GetFeedbackDataByIdAsync(recordId);
            if (record == null) return;

            // Remove old LSH entries (clear from all hash tables for this ID)
            foreach (var hashTable in _hashTables)
            {
                foreach (var bucket in hashTable.Values)
                {
                    bucket.Remove(recordId);
                }
            }

            // Re-add with new embeddings
            if (record.RequestEmbedding != null)
                InsertLSH(recordId, record.RequestEmbedding);

            if (record.TextResponseEmbedding != null)
                InsertLSH(recordId, record.TextResponseEmbedding);

            if (record.ToolUseTextResponseEmbedding != null)
                InsertLSH(recordId, record.ToolUseTextResponseEmbedding);

            if (record.RequestEmbeddingList != null)
                foreach (var emb in record.RequestEmbeddingList)
                    InsertLSH(recordId, emb);

            if (record.TextResponseEmbeddingList != null)
                foreach (var emb in record.TextResponseEmbeddingList)
                    InsertLSH(recordId, emb);

            if (record.ToolUseTextResponseEmbeddingList != null)
                foreach (var emb in record.ToolUseTextResponseEmbeddingList)
                    InsertLSH(recordId, emb);
        }

        /// <summary>
        /// Clears all LSH tables and rebuilds from database
        /// Call this after major embedding changes
        /// </summary>
        public async Task RebuildAllLSHAsync(Action<int, int>? progress = null)
        {
            LogMessage("Rebuilding all LSH indexes...");

            // Clear all hash tables
            foreach (var hashTable in _hashTables)
            {
                hashTable.Clear();
            }

            // Reload from database
            const string query = "SELECT Id, RequestEmbedding, TextResponseEmbedding, ToolUseTextResponseEmbedding, " +
                                "RequestEmbeddingList, TextResponseEmbeddingList, ToolUseTextResponseEmbeddingList " +
                                "FROM embeddings WHERE RequestEmbedding IS NOT NULL";

            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand(query, connection);

            var records = new List<(int Id, string? RE, string? TRE, string? TUTRE, string? REL, string? TREL, string? TUTREL)>();

            using (var reader = await command.ExecuteReaderAsync())
            {
                while (await reader.ReadAsync())
                {
                    records.Add((
                        reader.GetInt32(0),
                        reader.IsDBNull(1) ? null : reader.GetString(1),
                        reader.IsDBNull(2) ? null : reader.GetString(2),
                        reader.IsDBNull(3) ? null : reader.GetString(3),
                        reader.IsDBNull(4) ? null : reader.GetString(4),
                        reader.IsDBNull(5) ? null : reader.GetString(5),
                        reader.IsDBNull(6) ? null : reader.GetString(6)
                    ));
                }
            }

            int processed = 0;
            foreach (var record in records)
            {
                if (!string.IsNullOrEmpty(record.RE))
                {
                    var emb = JsonConvert.DeserializeObject<float[]>(record.RE);
                    if (emb != null) InsertLSH(record.Id, emb);
                }

                if (!string.IsNullOrEmpty(record.TRE))
                {
                    var emb = JsonConvert.DeserializeObject<float[]>(record.TRE);
                    if (emb != null) InsertLSH(record.Id, emb);
                }

                if (!string.IsNullOrEmpty(record.TUTRE))
                {
                    var emb = JsonConvert.DeserializeObject<float[]>(record.TUTRE);
                    if (emb != null) InsertLSH(record.Id, emb);
                }

                if (!string.IsNullOrEmpty(record.REL))
                {
                    var list = JsonConvert.DeserializeObject<List<float[]>>(record.REL);
                    if (list != null) foreach (var emb in list) InsertLSH(record.Id, emb);
                }

                if (!string.IsNullOrEmpty(record.TREL))
                {
                    var list = JsonConvert.DeserializeObject<List<float[]>>(record.TREL);
                    if (list != null) foreach (var emb in list) InsertLSH(record.Id, emb);
                }

                if (!string.IsNullOrEmpty(record.TUTREL))
                {
                    var list = JsonConvert.DeserializeObject<List<float[]>>(record.TUTREL);
                    if (list != null) foreach (var emb in list) InsertLSH(record.Id, emb);
                }

                processed++;
                progress?.Invoke(processed, records.Count);
            }

            LogMessage($"LSH rebuild complete. Processed {processed} records.");
        }

        #endregion

        #region Backfill Result Class

        public class BackfillResult
        {
            public int TotalRecords { get; set; }
            public int ProcessedCount { get; set; }
            public int SuccessCount { get; set; }
            public int FailedCount { get; set; }
            public int SkippedCount { get; set; }
            public bool WasCancelled { get; set; }
            public TimeSpan ElapsedTime { get; set; }
            public List<string> Errors { get; set; } = new List<string>();

            public override string ToString() =>
                $"Backfill: {SuccessCount}/{TotalRecords} succeeded, {FailedCount} failed, {SkippedCount} skipped in {ElapsedTime.TotalSeconds:F1}s" +
                (WasCancelled ? " (CANCELLED)" : "");
        }

        #endregion

        #region Mock Data Generation

        /// <summary>
        /// Populates the database with mock conversation data for testing.
        /// </summary>
        /// <param name="count">Number of mock conversations to generate</param>
        /// <param name="generateEmbeddings">Whether to generate embeddings for the mock data</param>
        /// <param name="progress">Optional progress callback</param>
        public async Task<int> PopulateWithMockDataAsync(
            int count = 50,
            bool generateEmbeddings = true,
            Action<int, int>? progress = null)
        {
            LogMessage($"Generating {count} mock conversations...");
            var mockData = GenerateMockConversations(count);
            int created = 0;

            foreach (var data in mockData)
            {
                try
                {
                    await AddDataToEmbeddingDatabaseAsync(data);
                    created++;
                    progress?.Invoke(created, count);
                }
                catch (Exception ex)
                {
                    LogMessage($"Error creating mock record: {ex.Message}");
                }
            }

            LogMessage($"Created {created} mock records");
            return created;
        }

        /// <summary>
        /// Generates a list of mock conversation data
        /// </summary>
        private List<FeedbackDatabaseValues> GenerateMockConversations(int count)
        {
            var conversations = new List<FeedbackDatabaseValues>();
            var random = new Random();

            // Programming topics for realistic RAG testing
            var programmingQuestions = new[]
            {
                ("How do I read a file in C#?", "You can use File.ReadAllText() or StreamReader to read files in C#. For async operations, use File.ReadAllTextAsync(). Example: string content = await File.ReadAllTextAsync(\"path/to/file.txt\");"),
                ("What is dependency injection?", "Dependency Injection (DI) is a design pattern where objects receive their dependencies from external sources rather than creating them internally. This promotes loose coupling and makes code more testable. In .NET, you can use the built-in IServiceCollection."),
                ("Explain async await in C#", "Async/await is a pattern for asynchronous programming. The async keyword marks a method as asynchronous, and await pauses execution until the awaited task completes without blocking the thread. This improves application responsiveness."),
                ("How to connect to a database?", "Use ADO.NET with SqlConnection or an ORM like Entity Framework. Example: using var connection = new SqlConnection(connectionString); await connection.OpenAsync(); Then execute commands with SqlCommand."),
                ("What is LINQ?", "LINQ (Language Integrated Query) provides a consistent query syntax for collections, databases, and XML. It includes methods like Where(), Select(), OrderBy(), and GroupBy(). Example: var results = list.Where(x => x.Age > 18).ToList();"),
                ("How to handle exceptions?", "Use try-catch-finally blocks. Catch specific exceptions before general ones. Use 'throw;' to rethrow without losing stack trace. Consider using custom exception types for domain-specific errors."),
                ("What are generics in C#?", "Generics allow you to write type-safe code that works with any data type. Use angle brackets: List<T>, Dictionary<TKey, TValue>. They provide compile-time type checking and avoid boxing/unboxing overhead."),
                ("How to use Entity Framework?", "Install Microsoft.EntityFrameworkCore, create a DbContext with DbSet properties for your entities, configure the connection string, and use methods like Add(), SaveChanges(), and LINQ queries to interact with the database."),
                ("What is the difference between interface and abstract class?", "Interfaces define contracts with no implementation (until C# 8 default methods). Abstract classes can have implementation and state. A class can implement multiple interfaces but inherit only one class. Use interfaces for 'can-do' relationships."),
                ("How to create a REST API?", "Use ASP.NET Core Web API. Create controllers inheriting from ControllerBase, decorate with [ApiController] and [Route] attributes. Use [HttpGet], [HttpPost], etc. for endpoints. Return ActionResult<T> for proper HTTP responses."),
                ("What is middleware in ASP.NET?", "Middleware are components that handle HTTP requests/responses in a pipeline. Each middleware can process the request, pass it to the next component, and optionally process the response. Common examples: authentication, logging, exception handling."),
                ("How to implement caching?", "Use IMemoryCache for in-memory caching or IDistributedCache for distributed scenarios (Redis, SQL Server). Implement cache-aside pattern: check cache first, fetch from source if missing, store in cache. Set appropriate expiration policies."),
                ("What is the repository pattern?", "Repository pattern abstracts data access logic into separate classes. It provides a collection-like interface for accessing domain objects. Benefits: testability, separation of concerns, swappable data sources. Often used with Unit of Work pattern."),
                ("How to write unit tests?", "Use a testing framework like xUnit, NUnit, or MSTest. Structure tests with Arrange-Act-Assert pattern. Use mocking libraries like Moq for dependencies. Aim for high code coverage but focus on meaningful tests."),
                ("What is SignalR?", "SignalR is a library for real-time web functionality. It enables server-side code to push content to clients instantly. Uses WebSockets when available, falls back to other techniques. Great for chat apps, live dashboards, notifications."),
            };

            var toolOperations = new[]
            {
                ("file_read", "Reading file: /src/Program.cs", "public class Program { public static void Main() { Console.WriteLine(\"Hello\"); } }"),
                ("file_write", "Writing to: /src/NewClass.cs", "File written successfully. 45 bytes written."),
                ("terminal", "Executing: dotnet build", "Build succeeded. 0 Warning(s) 0 Error(s) Time Elapsed 00:00:02.34"),
                ("terminal", "Executing: dotnet test", "Passed! - Failed: 0, Passed: 23, Skipped: 0, Total: 23"),
                ("search", "Searching for: async methods", "Found 15 matches in 8 files"),
                ("git", "git status", "On branch main. Changes not staged for commit: modified: src/Service.cs"),
                ("database", "SELECT * FROM Users WHERE Active = 1", "Returned 42 rows in 0.023 seconds"),
                ("api_call", "GET https://api.example.com/data", "Status: 200 OK. Response: {\"success\": true, \"count\": 100}"),
            };

            var summaries = new[]
            {
                "Explained file reading operations in C# with examples",
                "Discussed dependency injection principles and .NET implementation",
                "Covered async/await patterns for non-blocking operations",
                "Demonstrated database connectivity options",
                "Explained LINQ query syntax and common operations",
                "Reviewed exception handling best practices",
                "Introduced generics for type-safe collections",
                "Walked through Entity Framework setup and usage",
                "Compared interfaces vs abstract classes",
                "Created REST API endpoints with ASP.NET Core",
            };

            for (int i = 0; i < count; i++)
            {
                var questionPair = programmingQuestions[random.Next(programmingQuestions.Length)];
                var hasToolUse = random.NextDouble() > 0.5;
                var hasSummary = random.NextDouble() > 0.3;

                var data = new FeedbackDatabaseValues
                {
                    RequestID = Guid.NewGuid().ToString(),
                    UserMessageType = "text",
                    AssistantMessageType = hasToolUse ? "tool_use" : "text",
                    Request = questionPair.Item1 + (random.NextDouble() > 0.7 ? " Please provide an example." : ""),
                    TextResponse = questionPair.Item2,
                    Rating = random.Next(1, 6),
                    Embed = generateEmbeddings,
                    EmbedRequest = generateEmbeddings,
                    EmbedTextResponse = generateEmbeddings,
                    EmbedRequestList = generateEmbeddings,
                    EmbedTextResponseList = generateEmbeddings,
                };

                if (hasToolUse)
                {
                    var tool = toolOperations[random.Next(toolOperations.Length)];
                    data.ToolName = tool.Item1;
                    data.ToolContent = tool.Item2;
                    data.ToolResult = tool.Item3;
                    data.ToolUseTextResponse = $"I'll use the {tool.Item1} tool to help with this. {tool.Item2}";
                    data.EmbedToolUseTextResponse = generateEmbeddings;
                    data.EmbedToolUseTextResponseList = generateEmbeddings;
                }

                if (hasSummary)
                {
                    data.Summary = summaries[random.Next(summaries.Length)];
                    data.EmbedSummary = generateEmbeddings;
                    data.EmbedSummaryList = generateEmbeddings;
                }

                conversations.Add(data);
            }

            return conversations;
        }

        /// <summary>
        /// Clears all data from the embeddings table (use with caution!)
        /// </summary>
        public async Task ClearAllDataAsync()
        {
            LogMessage("Clearing all embedding data...");

            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand("DELETE FROM embeddings", connection);
            var deleted = await command.ExecuteNonQueryAsync();

            // Clear LSH tables
            foreach (var hashTable in _hashTables)
            {
                hashTable.Clear();
            }

            // Clear cache
            _cache.Clear();

            LogMessage($"Deleted {deleted} records");
        }

        /// <summary>
        /// Gets database statistics
        /// </summary>
        public async Task<DatabaseStats> GetStatsAsync()
        {
            var stats = new DatabaseStats();

            using var connection = await GetConnectionAsync();

            // Total records
            using (var cmd = new SqliteCommand("SELECT COUNT(*) FROM embeddings", connection))
            {
                stats.TotalRecords = Convert.ToInt32(await cmd.ExecuteScalarAsync());
            }

            // Records with embeddings
            using (var cmd = new SqliteCommand("SELECT COUNT(*) FROM embeddings WHERE RequestEmbedding IS NOT NULL", connection))
            {
                stats.RecordsWithEmbeddings = Convert.ToInt32(await cmd.ExecuteScalarAsync());
            }

            // Records missing embeddings
            stats.RecordsMissingEmbeddings = stats.TotalRecords - stats.RecordsWithEmbeddings;

            // LSH bucket stats
            stats.TotalLSHBuckets = _hashTables.Sum(t => t.Count);
            stats.TotalLSHEntries = _hashTables.Sum(t => t.Values.Sum(b => b.Count));

            return stats;
        }

        public class DatabaseStats
        {
            public int TotalRecords { get; set; }
            public int RecordsWithEmbeddings { get; set; }
            public int RecordsMissingEmbeddings { get; set; }
            public int TotalLSHBuckets { get; set; }
            public int TotalLSHEntries { get; set; }

            public override string ToString() =>
                $"Records: {TotalRecords} total, {RecordsWithEmbeddings} with embeddings, {RecordsMissingEmbeddings} missing\n" +
                $"LSH: {TotalLSHBuckets} buckets, {TotalLSHEntries} entries";
        }

        #endregion

        #region Internal - Data Operations

        private async Task AddDataToEmbeddingDatabaseAsync(FeedbackDatabaseValues data)
        {
            Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Adding data for RequestID: {data.RequestID}");

            FormatDataFields(data);

            if (data.Embed && !string.IsNullOrEmpty(data.RequestID))
            {
                data = await EmbedResponse(data);
            }

            if (!string.IsNullOrEmpty(data.RequestID) && await RequestIdExistsAsync(data.RequestID))
            {
                Debug.WriteLine($"RequestID {data.RequestID} already exists. Skipping insertion.");
                return;
            }

            await InsertDataAndProcessLSH(data);
        }

        private async Task InsertDataAndProcessLSH(FeedbackDatabaseValues data)
        {
            const string insertQuery = @"
INSERT INTO embeddings (
    RequestID, UserMessageType, AssistantMessageType, Rating,
    MetaData, Request, TextResponse, ToolUseTextResponse, ToolName, ToolContent, ToolResult,
    Code, Summary, Comment, Errors,
    RequestStatus, RequestEmbedding, TextResponseEmbedding, ToolUseTextResponseEmbedding,
    SummaryEmbedding, RequestEmbeddingList, TextResponseEmbeddingList, ToolUseTextResponseEmbeddingList,
    SummaryEmbeddingList
) VALUES (
    @RequestID, @UserMessageType, @AssistantMessageType, @Rating,
    @MetaData, @Request, @TextResponse, @ToolUseTextResponse, @ToolName, @ToolContent, @ToolResult,
    @Code, @Summary, @Comment, @Errors,
    @RequestStatus, @RequestEmbedding, @TextResponseEmbedding, @ToolUseTextResponseEmbedding,
    @SummaryEmbedding, @RequestEmbeddingList, @TextResponseEmbeddingList, @ToolUseTextResponseEmbeddingList,
    @SummaryEmbeddingList
);
SELECT last_insert_rowid();";

            using var connection = await GetConnectionAsync();
            using var command = CreateParameterizedCommand(insertQuery, data, connection);

            var result = await command.ExecuteScalarAsync();
            data.Id = Convert.ToInt32(result);

            // Process LSH indexing
            ProcessLSHForData(data);
        }

        private void ProcessLSHForData(FeedbackDatabaseValues data)
        {
            if (data.EmbedRequest && data.RequestEmbedding != null)
                InsertLSH(data.Id, data.RequestEmbedding);

            if (data.EmbedTextResponse && data.TextResponseEmbedding != null)
                InsertLSH(data.Id, data.TextResponseEmbedding);

            if (data.EmbedToolUseTextResponse && data.ToolUseTextResponseEmbedding != null)
                InsertLSH(data.Id, data.ToolUseTextResponseEmbedding);

            if (data.EmbedRequestList && data.RequestEmbeddingList != null)
                foreach (var embedding in data.RequestEmbeddingList)
                    InsertLSH(data.Id, embedding);

            if (data.EmbedTextResponseList && data.TextResponseEmbeddingList != null)
                foreach (var embedding in data.TextResponseEmbeddingList)
                    InsertLSH(data.Id, embedding);

            if (data.EmbedToolUseTextResponseList && data.ToolUseTextResponseEmbeddingList != null)
                foreach (var embedding in data.ToolUseTextResponseEmbeddingList)
                    InsertLSH(data.Id, embedding);
        }

        private async Task UpdateDataInEmbeddingDatabaseAsyncInternal(FeedbackDatabaseValues data)
        {
            Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Updating RequestID: {data.RequestID}");

            if (data.Embed && !string.IsNullOrEmpty(data.RequestID))
            {
                var embeddingStatus = await CheckExistingEmbeddingsAsync(data.RequestID);
                if (embeddingStatus.hasNullEmbeddings)
                    data = await EmbedResponse(data);
            }

            FormatDataFields(data);
            await UpdateDatabaseAsync(data);
        }

        private async Task UpdateDatabaseAsync(FeedbackDatabaseValues data)
        {
            const string query = @"
UPDATE embeddings SET
    UserMessageType = COALESCE(@UserMessageType, UserMessageType),
    AssistantMessageType = COALESCE(@AssistantMessageType, AssistantMessageType),
    Rating = CASE WHEN @Rating IS NOT NULL THEN @Rating ELSE Rating END,
    MetaData = COALESCE(@MetaData, MetaData),
    Request = COALESCE(@Request, Request),
    TextResponse = CASE WHEN @TextResponse IS NOT NULL THEN COALESCE(TextResponse, '') || @TextResponse ELSE TextResponse END,
    ToolUseTextResponse = CASE WHEN @ToolUseTextResponse IS NOT NULL THEN COALESCE(ToolUseTextResponse, '') || @ToolUseTextResponse ELSE ToolUseTextResponse END,
    ToolName = COALESCE(@ToolName, ToolName),
    ToolContent = CASE WHEN @ToolContent IS NOT NULL THEN COALESCE(ToolContent, '') || @ToolContent ELSE ToolContent END,
    ToolResult = CASE WHEN @ToolResult IS NOT NULL THEN COALESCE(ToolResult, '') || @ToolResult ELSE ToolResult END,
    Code = COALESCE(@Code, Code),
    Summary = COALESCE(@Summary, Summary),
    Comment = CASE WHEN @Comment IS NOT NULL THEN COALESCE(Comment, '') || @Comment ELSE Comment END,
    Errors = CASE WHEN @Errors IS NOT NULL THEN COALESCE(Errors, '') || @Errors ELSE Errors END,
    RequestStatus = COALESCE(@RequestStatus, RequestStatus),
    RequestEmbedding = COALESCE(@RequestEmbedding, RequestEmbedding),
    TextResponseEmbedding = COALESCE(@TextResponseEmbedding, TextResponseEmbedding),
    ToolUseTextResponseEmbedding = COALESCE(@ToolUseTextResponseEmbedding, ToolUseTextResponseEmbedding),
    SummaryEmbedding = COALESCE(@SummaryEmbedding, SummaryEmbedding),
    RequestEmbeddingList = COALESCE(@RequestEmbeddingList, RequestEmbeddingList),
    TextResponseEmbeddingList = COALESCE(@TextResponseEmbeddingList, TextResponseEmbeddingList),
    ToolUseTextResponseEmbeddingList = COALESCE(@ToolUseTextResponseEmbeddingList, ToolUseTextResponseEmbeddingList),
    SummaryEmbeddingList = COALESCE(@SummaryEmbeddingList, SummaryEmbeddingList)
WHERE RequestID = @RequestID";

            using var connection = await GetConnectionAsync();
            using var command = CreateParameterizedCommand(query, data, connection);

            int retryCount = 0;
            while (true)
            {
                try
                {
                    await command.ExecuteNonQueryAsync();
                    break;
                }
                catch (SqliteException ex) when (
                    (ex.SqliteErrorCode == 5 || // SQLITE_BUSY
                     ex.SqliteErrorCode == 6) && // SQLITE_LOCKED
                    retryCount < _config.MaxRetryAttempts)
                {
                    retryCount++;
                    await Task.Delay(_config.RetryDelayMs * (int)Math.Pow(2, retryCount - 1));
                }
            }
        }

        private SqliteCommand CreateParameterizedCommand(string query, FeedbackDatabaseValues data, SqliteConnection connection)
        {
            var command = new SqliteCommand(query, connection);

            void AddParam(string name, object? value) =>
                command.Parameters.AddWithValue(name, value ?? DBNull.Value);

            string? Serialize<T>(T? value) where T : class =>
                value != null ? JsonConvert.SerializeObject(value) : null;

            AddParam("@RequestID", data.RequestID);
            AddParam("@UserMessageType", data.UserMessageType);
            AddParam("@AssistantMessageType", data.AssistantMessageType);
            AddParam("@Rating", data.Rating);
            AddParam("@MetaData", data.MetaData);
            AddParam("@Request", data.Request);
            AddParam("@TextResponse", data.newTextResponse ?? data.TextResponse);
            AddParam("@ToolUseTextResponse", data.newToolUseTextResponse ?? data.ToolUseTextResponse);
            AddParam("@ToolName", data.newToolName ?? data.ToolName);
            AddParam("@ToolContent", data.newToolContent ?? data.ToolContent);
            AddParam("@ToolResult", data.newToolResult ?? data.ToolResult);
            AddParam("@Code", data.Code);
            AddParam("@Summary", data.Summary);
            AddParam("@Comment", data.newComment ?? data.Comment);
            AddParam("@Errors", data.Errors);
            AddParam("@RequestStatus", data.RequestStatus);
            AddParam("@RequestEmbedding", Serialize(data.RequestEmbedding));
            AddParam("@TextResponseEmbedding", Serialize(data.TextResponseEmbedding));
            AddParam("@ToolUseTextResponseEmbedding", Serialize(data.ToolUseTextResponseEmbedding));
            AddParam("@SummaryEmbedding", Serialize(data.SummaryEmbedding));
            AddParam("@RequestEmbeddingList", Serialize(data.RequestEmbeddingList));
            AddParam("@TextResponseEmbeddingList", Serialize(data.TextResponseEmbeddingList));
            AddParam("@ToolUseTextResponseEmbeddingList", Serialize(data.ToolUseTextResponseEmbeddingList));
            AddParam("@SummaryEmbeddingList", Serialize(data.SummaryEmbeddingList));

            command.CommandTimeout = 30;
            return command;
        }

        #endregion

        #region Internal - Search Operations

        private async Task<List<FeedbackDatabaseValues>> SearchWithFTSAsync(string searchText, int topK, int searchLevel)
        {
            var results = new List<FeedbackDatabaseValues>();
            var searchWords = PrepareSearchWords(searchText);

            if (searchWords.Length == 0)
                return results;

            // Build FTS5 query
            var ftsQuery = string.Join(" OR ", searchWords);

            const string query = @"
SELECT e.* FROM embeddings e
INNER JOIN embeddings_fts fts ON e.Id = fts.rowid
WHERE embeddings_fts MATCH @SearchText
LIMIT @TopK";

            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand(query, connection);
            command.Parameters.AddWithValue("@SearchText", ftsQuery);
            command.Parameters.AddWithValue("@TopK", topK * 2); // Get more than needed for filtering

            using var reader = await command.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                var feedbackData = ReadFeedbackFromReader(reader);

                bool isRequestValid = !string.IsNullOrEmpty(feedbackData.Request);
                bool isResponseValid = !string.IsNullOrEmpty(feedbackData.TextResponse);
                bool isToolResponseValid = !string.IsNullOrEmpty(feedbackData.ToolUseTextResponse);

                if ((searchLevel == 1 && isRequestValid) ||
                    (searchLevel == 2 && isRequestValid && isResponseValid) ||
                    (searchLevel == 3 && isRequestValid && isResponseValid && isToolResponseValid))
                {
                    var requestScore = CalculateWordMatchScore(searchWords, feedbackData.Request ?? "");
                    var responseScore = isResponseValid ? CalculateWordMatchScore(searchWords, feedbackData.TextResponse!) : 0;
                    var toolResponseScore = isToolResponseValid ? CalculateWordMatchScore(searchWords, feedbackData.ToolUseTextResponse!) : 0;

                    feedbackData.Similarity = searchLevel switch
                    {
                        1 => requestScore,
                        2 => Math.Max(requestScore, responseScore),
                        3 => Math.Max(requestScore, Math.Max(responseScore, toolResponseScore)),
                        _ => 0
                    };

                    results.Add(feedbackData);
                }
            }

            return results.OrderByDescending(x => x.Similarity).Take(topK).ToList();
        }

        private async Task<List<FeedbackDatabaseValues>> SearchWithLSHAsync(string searchText, int topK, float minimumSimilarity)
        {
            var results = new List<FeedbackDatabaseValues>();

            float[]? searchEmbedding = await _embedder.GetEmbeddingsAsync(searchText);
            if (searchEmbedding == null || searchEmbedding.Length == 0)
                return results;

            var candidateIds = SearchLSH(searchEmbedding, topK);

            foreach (var id in candidateIds)
            {
                var doc = await GetFeedbackDataByIdAsync(id);
                if (doc?.RequestEmbedding != null)
                {
                    doc.Similarity = CalculateCosineSimilarity(searchEmbedding, doc.RequestEmbedding);
                    if (doc.Similarity >= minimumSimilarity)
                        results.Add(doc);
                }
            }

            return results.OrderByDescending(x => x.Similarity).Take(topK).ToList();
        }

        private async Task<List<FeedbackDatabaseValues>> SearchByTextContentAsync(string text)
        {
            var results = new List<FeedbackDatabaseValues>();

            const string query = @"
SELECT * FROM embeddings
WHERE Request LIKE @Text
   OR TextResponse LIKE @Text
   OR ToolUseTextResponse LIKE @Text
LIMIT 100";

            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand(query, connection);
            command.Parameters.AddWithValue("@Text", $"%{text.Trim()}%");

            using var reader = await command.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                results.Add(ReadFeedbackFromReader(reader));
            }

            return results;
        }

        #endregion

        #region LSH Operations

        private void InsertLSH(int id, float[] vector)
        {
            var hashCodes = ComputeLSH(vector);
            for (int i = 0; i < _config.NumberOfHashTables; i++)
            {
                int hashCode = hashCodes[i];
                if (!_hashTables[i].ContainsKey(hashCode))
                {
                    _hashTables[i][hashCode] = new List<int>();
                }
                _hashTables[i][hashCode].Add(id);
            }
        }

        private List<int> SearchLSH(float[] queryVector, int topK)
        {
            var hashCodes = ComputeLSH(queryVector);
            var candidateSet = new HashSet<int>();

            for (int i = 0; i < _config.NumberOfHashTables; i++)
            {
                if (_hashTables[i].TryGetValue(hashCodes[i], out var bucket))
                {
                    candidateSet.UnionWith(bucket);
                }
            }

            return candidateSet.Take(topK * 2).ToList();
        }

        private List<int> ComputeLSH(float[] vector)
        {
            var hashCodes = new List<int>();

            for (int i = 0; i < _config.NumberOfHashTables; i++)
            {
                int hashCode = 0;
                for (int j = 0; j < _config.NumberOfHashFunctions; j++)
                {
                    float dotProduct = 0;
                    var hashVector = _hashFunctions[i][j];
                    int minLength = Math.Min(vector.Length, hashVector.Length);

                    for (int k = 0; k < minLength; k++)
                        dotProduct += vector[k] * hashVector[k];

                    hashCode = (hashCode << 1) | (dotProduct > 0 ? 1 : 0);
                }
                hashCodes.Add(hashCode);
            }

            return hashCodes;
        }

        #endregion

        #region Embedding Operations

        private async Task<FeedbackDatabaseValues> EmbedResponse(FeedbackDatabaseValues data)
        {
            var tasks = new List<Task>();

            if (data.EmbedRequestList && !string.IsNullOrEmpty(data.Request))
                tasks.Add(_embedder.HandleListEmbedding(data.Request, true, list => data.RequestEmbeddingList = list));

            if (data.EmbedTextResponseList && !string.IsNullOrEmpty(data.TextResponse))
                tasks.Add(_embedder.HandleListEmbedding(data.TextResponse, true, list => data.TextResponseEmbeddingList = list));

            if (data.EmbedToolUseTextResponseList && !string.IsNullOrEmpty(data.ToolUseTextResponse))
                tasks.Add(_embedder.HandleListEmbedding(data.ToolUseTextResponse, true, list => data.ToolUseTextResponseEmbeddingList = list));

            if (data.EmbedSummaryList && !string.IsNullOrEmpty(data.Summary))
                tasks.Add(_embedder.HandleListEmbedding(data.Summary, true, list => data.SummaryEmbeddingList = list));

            if (data.EmbedRequest && !string.IsNullOrEmpty(data.Request))
                tasks.Add(_embedder.HandleSingleEmbedding(data.Request, true, emb => data.RequestEmbedding = emb));

            if (data.EmbedTextResponse && !string.IsNullOrEmpty(data.TextResponse))
                tasks.Add(_embedder.HandleSingleEmbedding(data.TextResponse, true, emb => data.TextResponseEmbedding = emb));

            if (data.EmbedToolUseTextResponse && !string.IsNullOrEmpty(data.ToolUseTextResponse))
                tasks.Add(_embedder.HandleSingleEmbedding(data.ToolUseTextResponse, true, emb => data.ToolUseTextResponseEmbedding = emb));

            if (data.EmbedSummary && !string.IsNullOrEmpty(data.Summary))
                tasks.Add(_embedder.HandleSingleEmbedding(data.Summary, true, emb => data.SummaryEmbedding = emb));

            if (tasks.Count > 0)
            {
                var timeoutTask = Task.Delay(TimeSpan.FromSeconds(30));
                var completedTask = await Task.WhenAny(Task.WhenAll(tasks), timeoutTask);

                if (completedTask == timeoutTask)
                {
                    LogMessage($"Warning: Embedding generation timed out for RequestID: {data.RequestID}");
                }
            }

            return data;
        }

        private async Task<(bool hasNullEmbeddings, EmbeddingStatus status)> CheckExistingEmbeddingsAsync(string requestId)
        {
            const string query = @"
SELECT RequestEmbedding, RequestEmbeddingList, 
       TextResponseEmbedding, TextResponseEmbeddingList,
       ToolUseTextResponseEmbedding, ToolUseTextResponseEmbeddingList,
       SummaryEmbedding, SummaryEmbeddingList
FROM embeddings WHERE RequestID = @RequestID";

            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand(query, connection);
            command.Parameters.AddWithValue("@RequestID", requestId);

            using var reader = await command.ExecuteReaderAsync();
            if (!await reader.ReadAsync())
                return (true, new EmbeddingStatus());

            var status = new EmbeddingStatus
            {
                RequestEmbedding = !reader.IsDBNull(0),
                RequestEmbeddingList = !reader.IsDBNull(1),
                TextResponseEmbedding = !reader.IsDBNull(2),
                TextResponseEmbeddingList = !reader.IsDBNull(3),
                ToolUseTextResponseEmbedding = !reader.IsDBNull(4),
                ToolUseTextResponseEmbeddingList = !reader.IsDBNull(5),
                SummaryEmbedding = !reader.IsDBNull(6),
                SummaryEmbeddingList = !reader.IsDBNull(7)
            };

            bool hasNull = Enumerable.Range(0, 8).Any(i => reader.IsDBNull(i));
            return (hasNull, status);
        }

        #endregion

        #region Similarity Calculations

        private double CalculateCosineSimilarity(float[] embedding1, float[] embedding2)
        {
            if (embedding1 == null || embedding2 == null || embedding1.Length != embedding2.Length)
                return 0;

            double dotProduct = 0, magnitude1 = 0, magnitude2 = 0;

            for (int i = 0; i < embedding1.Length; i++)
            {
                dotProduct += embedding1[i] * embedding2[i];
                magnitude1 += embedding1[i] * embedding1[i];
                magnitude2 += embedding2[i] * embedding2[i];
            }

            magnitude1 = Math.Sqrt(magnitude1);
            magnitude2 = Math.Sqrt(magnitude2);

            return (magnitude1 == 0 || magnitude2 == 0) ? 0 : dotProduct / (magnitude1 * magnitude2);
        }

        private double CalculateWordMatchScore(string[] searchWords, string documentText)
        {
            if (string.IsNullOrEmpty(documentText) || searchWords.Length == 0)
                return 0;

            documentText = documentText.ToLower();
            var docWords = documentText.Split(new[] { ' ', '.', ',', ';', ':', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
            var docWordSet = new HashSet<string>(docWords, StringComparer.OrdinalIgnoreCase);

            // Use word boundary matching instead of substring contains
            // This prevents "the" matching in "brother", "together", etc.
            int matchedWords = searchWords.Count(word => docWordSet.Contains(word));
            double baseScore = (double)matchedWords / searchWords.Length;

            double phraseMatchBonus = documentText.Contains(string.Join(" ", searchWords)) ? 0.2 : 0;
            double density = docWords.Length > 0 ? (double)matchedWords / docWords.Length : 0;

            return Math.Min(baseScore + phraseMatchBonus + (density * 0.1), 1.0);
        }

        #endregion

        #region Helper Methods

        private async Task<bool> RequestIdExistsAsync(string requestId)
        {
            const string query = "SELECT COUNT(1) FROM embeddings WHERE RequestID = @RequestID";

            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand(query, connection);
            command.Parameters.AddWithValue("@RequestID", requestId);

            return Convert.ToInt32(await command.ExecuteScalarAsync()) > 0;
        }

        private void FormatDataFields(FeedbackDatabaseValues data)
        {
            if (!string.IsNullOrEmpty(data.Comment))
                data.newComment = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}\n\n{data.Comment}\n\n----\n\n";

            if (!string.IsNullOrEmpty(data.ToolContent))
                data.newToolContent = $"{data.ToolName ?? "N/A"}\n{data.ToolContent}\n";

            if (!string.IsNullOrEmpty(data.ToolResult))
                data.newToolResult = $"{data.ToolName ?? "N/A"}\n{data.ToolResult}\n";

            if (!string.IsNullOrEmpty(data.TextResponse))
                data.newTextResponse = $"{data.TextResponse}\n";

            if (!string.IsNullOrEmpty(data.ToolUseTextResponse))
                data.newToolUseTextResponse = $"{data.ToolName ?? "N/A"}\n{data.ToolUseTextResponse}\n";

            if (!string.IsNullOrEmpty(data.ToolName))
                data.newToolName = $"{data.ToolName}\n";
        }

        private string[] PrepareSearchWords(string searchText) =>
            searchText.ToLower()
                .Split(new[] { ' ', '.', ',', ';', ':', '!', '?' }, StringSplitOptions.RemoveEmptyEntries)
                .Where(word => !StopWords.Contains(word) && word.Length > 1)
                .ToArray();

        private FeedbackDatabaseValues ReadFeedbackFromReader(DbDataReader reader)
        {
            var data = new FeedbackDatabaseValues();

            string? GetString(string column)
            {
                try
                {
                    var ordinal = reader.GetOrdinal(column);
                    return reader.IsDBNull(ordinal) ? null : reader.GetString(ordinal);
                }
                catch { return null; }
            }

            int? GetInt(string column)
            {
                try
                {
                    var ordinal = reader.GetOrdinal(column);
                    return reader.IsDBNull(ordinal) ? null : reader.GetInt32(ordinal);
                }
                catch { return null; }
            }

            DateTime? GetDateTime(string column)
            {
                try
                {
                    var ordinal = reader.GetOrdinal(column);
                    return reader.IsDBNull(ordinal) ? null : reader.GetDateTime(ordinal);
                }
                catch { return null; }
            }

            T? Deserialize<T>(string column) where T : class
            {
                var json = GetString(column);
                return string.IsNullOrEmpty(json) ? null : JsonConvert.DeserializeObject<T>(json);
            }

            data.Id = GetInt("Id") ?? 0;
            data.RequestID = GetString("RequestID");
            data.UserMessageType = GetString("UserMessageType");
            data.AssistantMessageType = GetString("AssistantMessageType");
            data.Rating = GetInt("Rating") ?? 0;
            data.MetaData = GetString("MetaData");
            data.Request = GetString("Request");
            data.TextResponse = GetString("TextResponse");
            data.ToolUseTextResponse = GetString("ToolUseTextResponse");
            data.ToolName = GetString("ToolName");
            data.ToolContent = GetString("ToolContent");
            data.ToolResult = GetString("ToolResult");
            data.Code = GetString("Code");
            data.Summary = GetString("Summary");
            data.Comment = GetString("Comment");
            data.Errors = GetString("Errors");
            data.RequestStatus = GetString("RequestStatus");
            data.Timestamp = GetDateTime("Timestamp");

            data.RequestEmbedding = Deserialize<float[]>("RequestEmbedding");
            data.RequestEmbeddingList = Deserialize<List<float[]>>("RequestEmbeddingList");
            data.TextResponseEmbedding = Deserialize<float[]>("TextResponseEmbedding");
            data.TextResponseEmbeddingList = Deserialize<List<float[]>>("TextResponseEmbeddingList");
            data.ToolUseTextResponseEmbedding = Deserialize<float[]>("ToolUseTextResponseEmbedding");
            data.ToolUseTextResponseEmbeddingList = Deserialize<List<float[]>>("ToolUseTextResponseEmbeddingList");
            data.SummaryEmbedding = Deserialize<float[]>("SummaryEmbedding");
            data.SummaryEmbeddingList = Deserialize<List<float[]>>("SummaryEmbeddingList");

            return data;
        }

        private void InvalidateCache(string requestId)
        {
            _cache.TryRemove($"feedback_{requestId}", out _);
            var relatedKeys = _cache.Keys.Where(k => k.Contains(requestId)).ToList();
            foreach (var key in relatedKeys)
                _cache.TryRemove(key, out _);
        }

        private void CleanupCache()
        {
            var expired = _cache.Where(kvp => kvp.Value.Expiry <= DateTime.UtcNow).ToList();
            foreach (var item in expired)
                _cache.TryRemove(item.Key, out _);
        }

        private void LogMessage(string message)
        {
            _logBuilder.AppendLine($"{DateTime.Now:HH:mm:ss.fff} {message}");
            Debug.WriteLine(message);
        }

        #endregion

        #region Background Processing

        private void StartProcessor()
        {
            _processingTask = Task.Run(async () =>
            {
                while (!_cancellationTokenSource.Token.IsCancellationRequested)
                {
                    if (_updateQueue.TryDequeue(out var item))
                    {
                        await _processingSemaphore.WaitAsync(_cancellationTokenSource.Token);
                        try
                        {
                            await UpdateDataInEmbeddingDatabaseAsyncInternal(item.data);
                            item.completion.TrySetResult(true);
                        }
                        catch (Exception ex)
                        {
                            item.completion.TrySetException(ex);
                            LogMessage($"Error processing queue item: {ex.Message}");
                        }
                        finally
                        {
                            _processingSemaphore.Release();
                        }
                    }
                    else
                    {
                        await Task.Delay(100, _cancellationTokenSource.Token);
                    }
                }
            }, _cancellationTokenSource.Token);
        }

        #endregion

        #region Queue Status

        public int GetQueueLength() => _updateQueue.Count;

        #endregion

        #region Disposal

        public void Dispose()
        {
            if (_isDisposing) return;
            _isDisposing = true;

            _cancellationTokenSource.Cancel();

            try
            {
                _processingTask?.Wait(TimeSpan.FromSeconds(5));
            }
            catch { }

            _cleanupTimer?.Dispose();
            _cache.Clear();
            _processingSemaphore.Dispose();
            _queueSemaphore.Dispose();
            _cancellationTokenSource.Dispose();
            _embedder?.Dispose();
        }

        public async Task DisposeAsync()
        {
            if (_isDisposing) return;
            _isDisposing = true;

            _cancellationTokenSource.Cancel();

            try
            {
                await Task.WhenAny(_processingTask ?? Task.CompletedTask, Task.Delay(TimeSpan.FromSeconds(30)));
            }
            catch { }

            _cleanupTimer?.Dispose();
            _cache.Clear();
            _processingSemaphore.Dispose();
            _queueSemaphore.Dispose();
            _cancellationTokenSource.Dispose();
            _embedder?.Dispose();
        }

        #endregion

        #region Inner Classes

        private class CacheEntry<T>
        {
            public T Data { get; set; } = default!;
            public DateTime Expiry { get; set; }
            public long Size { get; set; }
        }

        private record EmbeddingStatus
        {
            public bool RequestEmbedding { get; init; }
            public bool RequestEmbeddingList { get; init; }
            public bool TextResponseEmbedding { get; init; }
            public bool TextResponseEmbeddingList { get; init; }
            public bool ToolUseTextResponseEmbedding { get; init; }
            public bool ToolUseTextResponseEmbeddingList { get; init; }
            public bool SummaryEmbedding { get; init; }
            public bool SummaryEmbeddingList { get; init; }
        }

        #endregion
    }

    #region Supporting Types (if not defined elsewhere)

 

 

    #endregion
}