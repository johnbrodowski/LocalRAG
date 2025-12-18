using Microsoft.ML.OnnxRuntime;

using Newtonsoft.Json;

using System.Collections.Concurrent;
using System.Data;
using System.Data.Common;
using Microsoft.Data.Sqlite;
using System.Diagnostics;
using System.Text;

namespace AnthropicApp.Database
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
                    var vector = new float[_config.MaxSequenceLength];
                    for (int k = 0; k < _config.MaxSequenceLength; k++)
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

            int matchedWords = searchWords.Count(word => documentText.Contains(word));
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