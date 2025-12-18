using AnthropicApp.Database;

using Microsoft.ML.OnnxRuntime;

using Newtonsoft.Json;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Data;
using System.Data.Common;
using Microsoft.Data.Sqlite;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnthropicApp.Database
{
    public partial class EmbeddingDatabaseOriginal
    {
        private const int MAX_QUEUE_SIZE = 1000; // Maximum queue size
        private const int MAX_RETRY_ATTEMPTS = 3;
        private const int RETRY_DELAY_MS = 1000;
        private readonly SemaphoreSlim _queueSemaphore;
        private bool _isDisposing;
        private readonly object _processLock = new object();


        //Caching parameters

        private readonly ConcurrentDictionary<string, CacheEntry<object>> _cache = new();

        //private readonly ConcurrentDictionary<string, CacheEntry<object>> _cache = new();
        //private readonly ConcurrentDictionary<string, (object data, DateTime expiry)> _cache = new();
        private readonly MemoryCacheOptions _cacheOptions;

        private readonly TimeSpan _defaultExpiry = TimeSpan.FromMinutes(15);
        private readonly System.Threading.Timer _cleanupTimer;
        // Add these to your ConcurrentDictionary declaration



        private readonly ConcurrentQueue<(FeedbackDatabaseValues data, TaskCompletionSource<bool> completion)> _updateQueue;
        private readonly SemaphoreSlim _processingSemaphore;
        private readonly CancellationTokenSource _cancellationTokenSource;
        private Task _processingTask;
        private bool _isProcessing;




        //Embedding parameters must match embedder class
        private const int MAX_SEQUENCE_LENGTH = 512;

        //LSH parameters
        private const int NUMBER_OF_HASH_FUNCTIONS = 8;

        private const int NUMBER_OF_HASH_TABLES = 10;

        //SQLite parameters
        // private string _connectionString = "data source=Database\\Memory\\FeedbackEmbeddings512.db";
        private string _connectionString = $"Data Source=Database\\Memory\\FeedbackEmbeddings512.db;Pooling=True;";

        private SqliteConnection? _connection;
        private StringBuilder _logBuilder;

        // Changed from 512 to match the existing MAX_SEQUENCE_LENGTH
        private List<List<float[]>> hashFunctions;

        private List<Dictionary<int, List<int>>> hashTables;

        // Pre-fetch the ordinals
        private int idIndex;

        private int requestIdIndex;
        private int userMessageTypeIndex;
        private int assistantMessageTypeIndex;
        private int ratingIndex;
        private int metaDataIndex;
        private int requestIndex;
        private int textResponseIndex;
        private int toolUseTextResponseIndex;
        private int toolNameIndex;
        private int toolContentIndex;
        private int toolResultIndex;
        private int codeIndex;
        private int summaryIndex;
        private int commentIndex;
        private int errorsIndex;
        private int requestStatusIndex;
        private int requestEmbeddingIndex;
        private int requestEmbeddingListIndex;

        private int textResponseEmbeddingIndex;
        private int textResponseEmbeddingListIndex;

        private int toolUseTextResponseEmbeddingIndex;
        private int toolUseTextResponseEmbeddingListIndex;

        private int summaryEmbeddingIndex;
        private int summaryEmbeddingListIndex;
        private int timestampIndex;
        private int statusIndex;


        private readonly EmbedderClassNew _embedder;




        public EmbeddingDatabaseOriginal()
        {
            _embedder = new EmbedderClassNew(new RAGConfiguration());
            _logBuilder = new StringBuilder();
            hashFunctions = new List<List<float[]>>();
            hashTables = new List<Dictionary<int, List<int>>>();

            _queueSemaphore = new SemaphoreSlim(MAX_QUEUE_SIZE, MAX_QUEUE_SIZE);
            _isDisposing = false;

            // Initialize connection pool
            // SqliteConnection.CreateFile("Database\\Memory\\FeedbackEmbeddings512.db");
            //   _connectionString = $"data source=Database\\Memory\\FeedbackEmbeddings512.db;Version=3;Pooling=True;";



            _logBuilder = new StringBuilder();


            //_connection = new SqliteConnection(); // Initialize _connection to avoid null warnings
            hashFunctions = new List<List<float[]>>(); // Initialize hashFunctions to avoid null warnings
            hashTables = new List<Dictionary<int, List<int>>>(); // Initialize hashTables to avoid null warnings


            //Cache initialization
            _cacheOptions = new MemoryCacheOptions
            {
                MaxItems = 10000,
                ItemSizeThreshold = 1024 * 1024 // 1MB
            };

            _cleanupTimer = new System.Threading.Timer(
                _ => CleanupCache(),
                null,
                TimeSpan.FromMinutes(5),
                TimeSpan.FromMinutes(5)
            );



            _ = CreateTable();
            _ = GetOrdinals();
            InitializeLSH();

            _updateQueue = new ConcurrentQueue<(FeedbackDatabaseValues, TaskCompletionSource<bool>)>();
            _processingSemaphore = new SemaphoreSlim(1, 1);
            _cancellationTokenSource = new CancellationTokenSource();
            _isProcessing = false;
            _processingTask = Task.CompletedTask;

            // Start the background processor
            StartProcessor();
        }



        public async Task<FeedbackDatabaseValues?> GetFeedbackDataByRequestIdAsync(string requestID)
        {
            return await GetOrSetCache(
                $"feedback_{requestID}",
                () => GetFeedbackDataFromEmbeddingDatabaseByRequestIDAsync(requestID),
                _defaultExpiry
            );
        }

        //private void CleanupCache()
        //{
        //    var expired = _cache.Where(kvp => kvp.Value.expiry <= DateTime.UtcNow);
        //    foreach (var item in expired)
        //    {
        //        _cache.TryRemove(item.Key, out _);
        //    }
        //}


        public class MemoryCacheOptions
        {
            public int MaxItems { get; set; }
            public long ItemSizeThreshold { get; set; }
        }

        private void InvalidateCache(string requestId)
        {
            _cache.TryRemove($"feedback_{requestId}", out _);
        }

        public class CacheEntry<T>
        {
            public T Data { get; set; }
            public DateTime Expiry { get; set; }
            public long Size { get; set; }
        }

        private bool ShouldCache(object item)
        {
            if (_cache.Count >= _cacheOptions.MaxItems)
                return false;

            // Estimate size based on JSON serialization
            var size = JsonConvert.SerializeObject(item).Length * 2; // UTF16 chars

            // Check against size threshold
            return size <= _cacheOptions.ItemSizeThreshold;
        }

        private void CleanupCache()
        {
            var expired = _cache.Where(kvp => kvp.Value.Expiry <= DateTime.UtcNow);
            foreach (var item in expired)
            {
                _cache.TryRemove(item.Key, out _);
            }
        }

        private async Task<T> GetOrSetCache<T>(string key, Func<Task<T>> getter, TimeSpan expiry)
        {
            if (_cache.TryGetValue(key, out var cached) && cached.Expiry > DateTime.UtcNow)
                return (T)cached.Data;

            var data = await getter();

            if (ShouldCache(data))
            {
                var entry = new CacheEntry<object>
                {
                    Data = data,
                    Expiry = DateTime.UtcNow.Add(expiry),
                    Size = JsonConvert.SerializeObject(data).Length * 2
                };

                _cache.AddOrUpdate(key, entry, (_, _) => entry);
            }

            return data;
        }















        private void StartProcessor()
        {
            _processingTask = Task.Run(async () =>
            {
                while (!_cancellationTokenSource.Token.IsCancellationRequested)
                {
                    if (_updateQueue.TryDequeue(out var item))
                    {
                        try
                        {
                            await _processingSemaphore.WaitAsync(_cancellationTokenSource.Token);
                            _isProcessing = true;

                            try
                            {
                                await UpdateDataInEmbeddingDatabaseAsyncInternal(item.data);
                                item.completion.TrySetResult(true);  // Use TrySetResult instead of SetResult
                            }
                            catch (Exception ex)
                            {
                                item.completion.TrySetException(ex);  // Use TrySetException instead of SetException
                                Debug.WriteLine($"Error processing queue item: {ex.Message}");
                            }
                        }
                        finally
                        {
                            _isProcessing = false;
                            _processingSemaphore.Release();
                        }
                    }
                    else
                    {
                        // If queue is empty, wait a bit before checking again
                        await Task.Delay(100, _cancellationTokenSource.Token);
                    }
                }
            }, _cancellationTokenSource.Token);
        }







        // New public method that uses the queue
        public async Task UpdateDataInEmbeddingDatabaseAsync(FeedbackDatabaseValues data)
        {
            if (data.RequestID != null)
            {
                InvalidateCache(data.RequestID);
                await UpdateDatabase(data);
            }


            var completionSource = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            _updateQueue.Enqueue((data, completionSource));
            await completionSource.Task;  // Wait for the operation to complete
        }



        private async Task<SqliteConnection> GetConnectionAsync()
        {
            var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync();
            return connection;
        }

        private async Task UpdateDataInEmbeddingDatabaseAsyncInternal(FeedbackDatabaseValues data)
        {
            Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Starting update for RequestID: {data.RequestID}");

            if (data.Embed && data.RequestID != null)
            {
                Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Starting embedding check for RequestID: {data.RequestID}");
                var embeddingStatus = await CheckExistingEmbeddings(data.RequestID);
                Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Finished embedding check for RequestID: {data.RequestID}");

                if (embeddingStatus.hasNullEmbeddings)
                {
                    Debug.WriteLine($"\nOK\n");
                    Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Starting EmbedResponse for RequestID: {data.RequestID}");
                    data = await EmbedResponse(data);
                    Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Finished EmbedResponse for RequestID: {data.RequestID}");
                }
            }

            if (!string.IsNullOrEmpty(data.Comment))
            {
                data.newComment = FormatComment(data.Comment);
            }

            if (!string.IsNullOrEmpty(data.ToolContent))
            {
                data.newToolContent = FormatToolContent(data.ToolName ?? "", data.ToolContent);
            }

            if (!string.IsNullOrEmpty(data.ToolResult))
            {
                data.newToolResult = FormatToolResult(data.ToolName ?? "", data.ToolResult);
            }

            if (!string.IsNullOrEmpty(data.ToolUseTextResponse))
            {
                data.newToolUseTextResponse = FormatToolUseTextResponse(data.ToolName ?? "", data.ToolUseTextResponse);
            }


            if (!string.IsNullOrEmpty(data.ToolUseTextResponse))
            {
                data.newToolUseTextResponse = FormatToolUseTextResponse(data.ToolName ?? "", data.ToolUseTextResponse);
            }


            if (!string.IsNullOrEmpty(data.TextResponse))
            {
                data.newTextResponse = FormatTextResponse(data.TextResponse);
            }

            if (!string.IsNullOrEmpty(data.ToolName))
            {
                data.newToolName = FormatToolName(data.ToolName);
            }


            Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Starting database update for RequestID: {data.RequestID}");
            await UpdateDatabase(data);
            Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Finished database update for RequestID: {data.RequestID}");
        }





        #region Database Methods


        public async Task AddRequestToEmbeddingDatabaseAsync(string? RequestId, string theRequest, bool embed = false)
        {
            await Task.Run(async () =>
            {
                var data = new FeedbackDatabaseValues
                {
                    RequestID = RequestId,
                    UserMessageType = "text",
                    Request = theRequest,
                    Embed = embed,
                    EmbedRequest = embed,
                    EmbedRequestList = embed
                };
                await AddDataToEmbeddingDatabaseAsync(data);
            });
        }

        public async Task UpdateToolUseTextResponse(string RequestId, string Message, bool embed = false)
        {
            var data = new FeedbackDatabaseValues
            {
                RequestID = RequestId,
                ToolUseTextResponse = Message,
                UserMessageType = "tool_use",
                Embed = embed,
                EmbedToolUseTextResponse = embed,
                EmbedToolUseTextResponseList = embed
            };

            if (embed)
            {
                var embeddingStatus = await CheckExistingEmbeddings(RequestId, "tooluse");
                if (embeddingStatus.hasNullEmbeddings)
                {
                    data = await EmbedResponse(data);
                }
            }

            await UpdateDataInEmbeddingDatabaseAsync(data);
        }

        public async Task UpdateToolContent(string RequestId, string ToolContent)
        {
            //  await Task.Run(async () =>
            //  {
            var data = new FeedbackDatabaseValues
            {
                RequestID = RequestId,
                ToolContent = ToolContent
            };
            ///////////////////////////// await UpdateDataInEmbeddingDatabaseAsync(data);
            // });
        }

        public async Task UpdateTextResponse(string RequestId, string Message, bool embed = false)
        {
            var data = new FeedbackDatabaseValues
            {
                RequestID = RequestId,
                TextResponse = Message,
                Embed = embed,
                EmbedTextResponse = embed,
                EmbedTextResponseList = embed
            };

            if (embed)
            {
                var embeddingStatus = await CheckExistingEmbeddings(RequestId, "text");
                if (embeddingStatus.hasNullEmbeddings)
                {
                    data = await EmbedResponse(data);
                }
            }

            await UpdateDataInEmbeddingDatabaseAsync(data);
        }

        public async Task UpdateToolResult(string RequestId, string Message, bool embed = false)
        {
            var data = new FeedbackDatabaseValues
            {
                RequestID = RequestId,
                ToolResult = Message,
                Embed = embed,
                EmbedTextResponse = embed,
                EmbedTextResponseList = embed
            };

            if (embed)
            {
                var embeddingStatus = await CheckExistingEmbeddings(RequestId, "text");
                if (embeddingStatus.hasNullEmbeddings)
                {
                    data = await EmbedResponse(data);
                }
            }

            await UpdateDataInEmbeddingDatabaseAsync(data);
        }

        public async Task UpdateSummary(string RequestId, string Message, bool embed = false)
        {
            var data = new FeedbackDatabaseValues
            {
                RequestID = RequestId,
                Summary = Message,
                Embed = embed,
                EmbedSummary = embed,
                EmbedSummaryList = embed
            };

            if (embed)
            {
                var embeddingStatus = await CheckExistingEmbeddings(RequestId, "summary");
                if (embeddingStatus.hasNullEmbeddings)
                {
                    data = await EmbedResponse(data);
                }
            }

            await UpdateDataInEmbeddingDatabaseAsync(data);
        }

        private async Task AddDataToEmbeddingDatabaseAsync(FeedbackDatabaseValues data)
        {
            Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Starting to add data for RequestID: {data.RequestID}");


            if (!string.IsNullOrEmpty(data.Comment))
            {
                data.newComment = FormatComment(data.Comment);
            }

            if (!string.IsNullOrEmpty(data.ToolContent))
            {
                data.newToolContent = FormatToolContent(data.ToolName ?? "", data.ToolContent);
            }

            if (!string.IsNullOrEmpty(data.ToolResult))
            {
                data.newToolResult = FormatToolResult(data.ToolName ?? "", data.ToolResult);
            }

            if (!string.IsNullOrEmpty(data.TextResponse))
            {
                data.newTextResponse = FormatTextResponse(data.TextResponse);
            }

            if (!string.IsNullOrEmpty(data.ToolUseTextResponse))
            {
                data.newToolUseTextResponse = FormatToolUseTextResponse(data.ToolName ?? "", data.ToolUseTextResponse);
            }

            if (!string.IsNullOrEmpty(data.ToolName))
            {
                data.newToolName = FormatToolName(data.ToolName);
            }

            // Handle embeddings if requested
            if (data.Embed == true && !string.IsNullOrEmpty(data.RequestID))
            {
                data = await EmbedResponse(data);
            }

            // Check if RequestID already exists
            if (!string.IsNullOrEmpty(data.RequestID) && await RequestIDExistsAsync(data.RequestID))
            {
                Debug.WriteLine($"RequestID {data.RequestID} already exists. Skipping insertion.");
                return;
            }

            // Insert the data and handle LSH
            await InsertDataAndProcessLSH(data);
        }

        public async Task AddRequestToolResultToEmbeddingDatabaseAsync(string? RequestId, string theRequest, bool embed = false)
        {
            await Task.Run(async () =>
            {
                var data = new FeedbackDatabaseValues
                {
                    RequestID = RequestId,
                    UserMessageType = "tool_result",
                    Request = theRequest,
                    Embed = embed,
                    EmbedRequest = embed,
                    EmbedRequestList = embed
                };
                await AddDataToEmbeddingDatabaseAsync(data);
            });
        }
















        private async Task<(bool hasNullEmbeddings, EmbeddingStatus status)> CheckExistingEmbeddings(string requestId, string type = "all")
        {
            Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Checking embeddings for RequestID: {requestId}, Type: {type}");

            // First check if the record exists at all
            const string existsQuery = "SELECT COUNT(1) FROM embeddings WHERE RequestID = @RequestID";

            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync();

            // First check existence
            using (var existsCommand = new SqliteCommand(existsQuery, connection))
            {
                existsCommand.Parameters.AddWithValue("@RequestID", requestId);
                var count = Convert.ToInt32(await existsCommand.ExecuteScalarAsync());
                if (count == 0)
                {
                    Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Record does not exist yet for RequestID");
                    return (true, new EmbeddingStatus());
                }
            }

            // Build query based on type
            string embeddingsQuery = "SELECT ";
            switch (type.ToLower())
            {
                case "tooluse":
                    embeddingsQuery += "ToolUseTextResponseEmbedding, ToolUseTextResponseEmbeddingList";
                    break;

                case "text":
                    embeddingsQuery += "TextResponseEmbedding, TextResponseEmbeddingList";
                    break;

                case "request":
                    embeddingsQuery += "RequestEmbedding, RequestEmbeddingList";
                    break;

                case "summary":
                    embeddingsQuery += "SummaryEmbedding, SummaryEmbeddingList";
                    break;

                default:
                    embeddingsQuery += @"ToolUseTextResponseEmbedding, ToolUseTextResponseEmbeddingList,
                                TextResponseEmbedding, TextResponseEmbeddingList,
                                RequestEmbedding, RequestEmbeddingList,
                                SummaryEmbedding, SummaryEmbeddingList";
                    break;
            }
            embeddingsQuery += " FROM embeddings WHERE RequestID = @RequestID";

            using (var command = new SqliteCommand(embeddingsQuery, connection))
            {
                command.Parameters.AddWithValue("@RequestID", requestId);
                using var reader = await command.ExecuteReaderAsync();

                if (await reader.ReadAsync())
                {
                    var status = new EmbeddingStatus();
                    bool hasNull = false;

                    // Check embeddings based on type
                    switch (type.ToLower())
                    {
                        case "tooluse":
                            status.ToolUseTextResponseEmbedding = !reader.IsDBNull(0);
                            status.ToolUseTextResponseEmbeddingList = !reader.IsDBNull(1);
                            hasNull = reader.IsDBNull(0) || reader.IsDBNull(1);
                            break;

                        case "text":
                            status.TextResponseEmbedding = !reader.IsDBNull(0);
                            status.TextResponseEmbeddingList = !reader.IsDBNull(1);
                            hasNull = reader.IsDBNull(0) || reader.IsDBNull(1);
                            break;

                        case "request":
                            status.RequestEmbedding = !reader.IsDBNull(0);
                            status.RequestEmbeddingList = !reader.IsDBNull(1);
                            hasNull = reader.IsDBNull(0) || reader.IsDBNull(1);
                            break;

                        case "summary":
                            status.SummaryEmbedding = !reader.IsDBNull(0);
                            status.SummaryEmbeddingList = !reader.IsDBNull(1);
                            hasNull = reader.IsDBNull(0) || reader.IsDBNull(1);
                            break;

                        default:
                            status.ToolUseTextResponseEmbedding = !reader.IsDBNull(0);
                            status.ToolUseTextResponseEmbeddingList = !reader.IsDBNull(1);
                            status.TextResponseEmbedding = !reader.IsDBNull(2);
                            status.TextResponseEmbeddingList = !reader.IsDBNull(3);
                            status.RequestEmbedding = !reader.IsDBNull(4);
                            status.RequestEmbeddingList = !reader.IsDBNull(5);
                            status.SummaryEmbedding = !reader.IsDBNull(6);
                            status.SummaryEmbeddingList = !reader.IsDBNull(7);
                            hasNull = Enumerable.Range(0, 8).Any(i => reader.IsDBNull(i));
                            break;
                    }

                    return (hasNull, status);
                }

                Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} No record found in second check");
                return (true, new EmbeddingStatus());
            }
        }



        public async Task<FeedbackDatabaseValues?> GetFeedbackDataByIdAsync(int id)
        {
            return await Task.Run(() =>
            {
                var data = GetFeedbackDataDataFromEmbeddingDatabaseById(id);
                return data;
            });
        }

        // Add new method to search using memory-based approach
        public async Task<List<FeedbackDatabaseValues>> SearchMemoryBasedAsync(string searchText, int topK)
        {
            LogMessage($"Starting memory-based search for: {searchText}");

            var similarResults = await _embedder.SearchSimilarAsync(searchText, topK);
            var feedbackResults = new List<FeedbackDatabaseValues>();

            foreach (var result in similarResults)
            {
                // Get the original text from the result metadata
                string originalText = result.Metadata.Tags["text"];

                // Search the database for entries containing this text
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













        public async Task<List<(string request, string textResponse, string toolUseTextResponse, string toolContent, string toolResult, string requestID)>> GetConversationHistoryAsync(int messageCount = 100)
        {
            return await Task.Run(() =>
            {
                var recentInteractions = GetRecentConversationHistoryAsync(messageCount);
                return recentInteractions;
            });
        }

        // Update the embedding methods to use the new EmbedderClass
        private async Task<FeedbackDatabaseValues> EmbedResponse(FeedbackDatabaseValues data)
        {
            await Task.WhenAll(
                _embedder.HandleListEmbedding(data.Request, data.EmbedRequestList, list => data.RequestEmbeddingList = list),
                _embedder.HandleListEmbedding(data.TextResponse, data.EmbedTextResponseList, list => data.TextResponseEmbeddingList = list),
                _embedder.HandleListEmbedding(data.ToolUseTextResponse, data.EmbedToolUseTextResponseList, list => data.ToolUseTextResponseEmbeddingList = list),
                _embedder.HandleListEmbedding(data.Summary, data.EmbedSummaryList, list => data.SummaryEmbeddingList = list)
            );

            await Task.WhenAll(
                _embedder.HandleSingleEmbedding(data.Request, data.EmbedRequest, emb => data.RequestEmbedding = emb),
                _embedder.HandleSingleEmbedding(data.TextResponse, data.EmbedTextResponse, emb => data.TextResponseEmbedding = emb),
                _embedder.HandleSingleEmbedding(data.ToolUseTextResponse, data.EmbedToolUseTextResponse, emb => data.ToolUseTextResponseEmbedding = emb),
                _embedder.HandleSingleEmbedding(data.Summary, data.EmbedSummary, emb => data.SummaryEmbedding = emb)
            );

            return data;
        }

        public async Task<List<FeedbackDatabaseValues>> SearchSimilarEmbeddingsLSHAsync(string searchText, int topK, float minimumSimilarity, int searchLevel)
        {
            LogMessage($"Starting improved search for: {searchText}");

            // Prepare search words
            var searchWords = PrepareSearchWords(searchText);
            LogMessage($"Search words: {string.Join(", ", searchWords)}");

            var columns = new List<string> { "Request", "TextResponse", "ToolUseTextResponse" };

            var sqlBuilder = new StringBuilder("SELECT * FROM embeddings WHERE ");

            var parameters = new List<SqliteParameter>();

            for (int i = 0; i < searchWords.Length; i++)
            {
                if (i > 0) sqlBuilder.Append(" OR ");
                sqlBuilder.Append("(");

                for (int j = 0; j < columns.Count; j++)
                {
                    if (j > 0) sqlBuilder.Append(" OR ");
                    sqlBuilder.Append($"{columns[j]} LIKE @param{i}_{j}");
                    parameters.Add(new SqliteParameter($"@param{i}_{j}", $"%{searchWords[i]}%"));
                }

                sqlBuilder.Append(")");
            }

            var similarDocuments = new List<FeedbackDatabaseValues>();
            var connection = new SqliteConnection(_connectionString);

            if (connection.State != System.Data.ConnectionState.Open)
            {
                await connection.OpenAsync();
            }

            try
            {
                using (var command = new SqliteCommand(sqlBuilder.ToString(), connection))
                {
                    command.Parameters.AddRange(parameters.ToArray());

                    using (var reader = await command.ExecuteReaderAsync())
                    {
                        while (await reader.ReadAsync())
                        {
                            var feedbackData = feedbackDatabaseValues(reader);

                            bool isRequestValid = !string.IsNullOrEmpty(feedbackData.Request);
                            bool isResponseValid = !string.IsNullOrEmpty(feedbackData.TextResponse);
                            bool isToolResponseValid = !string.IsNullOrEmpty(feedbackData.ToolUseTextResponse);

                            if ((searchLevel == 1 && isRequestValid) ||
                                (searchLevel == 2 && isRequestValid && isResponseValid) ||
                                (searchLevel == 3 && isRequestValid && isResponseValid && isToolResponseValid))
                            {
                                var requestScore = CalculateWordMatchScore(searchWords, feedbackData.Request);
                                var responseScore = isResponseValid ? CalculateWordMatchScore(searchWords, feedbackData.TextResponse) : 0;
                                var toolResponseScore = isToolResponseValid ? CalculateWordMatchScore(searchWords, feedbackData.ToolUseTextResponse) : 0;

                                // Calculate similarity based on the selected search level
                                switch (searchLevel)
                                {
                                    case 1:
                                        feedbackData.Similarity = requestScore;
                                        break;

                                    case 2:
                                        feedbackData.Similarity = Math.Max(requestScore, responseScore);
                                        break;

                                    case 3:
                                        feedbackData.Similarity = Math.Max(requestScore, Math.Max(responseScore, toolResponseScore));
                                        break;
                                }

                                similarDocuments.Add(feedbackData);
                            }
                        }
                    }
                }

                Debug.WriteLine($"Initial similar documents: {similarDocuments.Count}");
                // if (similarDocuments.Count < topK)
                //  {
                float[] searchEmbedding = await _embedder.GetEmbeddingsAsync(searchText) ?? Array.Empty<float>();
                Debug.WriteLine($"Search embedding length: {searchEmbedding.Length}");
                //  if (searchEmbedding.Length > 0)
                //  {
                //  List<int> candidateIds = await SearchLSHAsync(searchEmbedding, topK - similarDocuments.Count);
                List<int> candidateIds = await SearchLSHAsync(searchEmbedding, topK);
                Debug.WriteLine($"Found candidate IDs: {string.Join(", ", candidateIds)}");
                // Rest of the code...


                foreach (var id in candidateIds)
                {
                    if (!similarDocuments.Any(d => d.Id == id))
                    {
                        var doc = await GetFeedbackDataDataFromEmbeddingDatabaseById(id);
                        if (doc?.RequestEmbedding != null)
                        {
                            doc.Similarity = CalculateSimilarity(searchEmbedding, doc.RequestEmbedding);
                            similarDocuments.Add(doc);
                        }
                    }
                }


                //   }
                //  }
                Debug.WriteLine($"Final similar documents: {similarDocuments.Count}");

                // If we have less than topK results, use LSH to find more
                //if (similarDocuments.Count < topK)
                //{
                //  //  float[] searchEmbedding = await _embedder.GetEmbeddingsAsync(searchText) ?? Array.Empty<float>();
                //    if (searchEmbedding.Length > 0)
                //    {
                //      //  List<int> candidateIds = await SearchLSHAsync(searchEmbedding, topK - similarDocuments.Count);

                //        foreach (var editor_id in candidateIds)
                //        {
                //            if (!similarDocuments.Any(d => d.editor_id == editor_id))
                //            {
                //                var doc = await GetFeedbackDataDataFromEmbeddingDatabaseById(editor_id);
                //                if (doc?.RequestEmbedding != null)
                //                {
                //                    doc.Similarity = CalculateSimilarity(searchEmbedding, doc.RequestEmbedding);
                //                    similarDocuments.Add(doc);
                //                }
                //            }
                //        }
                //    }
                //}
            }
            catch (Exception ex)
            {
                LogMessage($"Error during database query: {ex.Message}");
                throw;
            }
            finally
            {
                if (connection.State == ConnectionState.Open)
                {
                    await connection.CloseAsync();
                }
            }

            LogMessage($"Search completed. Found {similarDocuments.Count} similar documents.");
            //return similarDocuments
            //    .OrderByDescending(x => x.Similarity)
            //    .Take(topK)
            //    .ToList();

            return similarDocuments
                .Where(x => x.Similarity >= minimumSimilarity)  // Filter by minimum similarity
                .OrderByDescending(x => x.Similarity)           // Order by similarity
                .Take(topK)                                     // Take top K results
                .ToList();                                      // Convert to list
        }


        #endregion Database Methods


        #region LSH

        private void InitializeLSH()
        {
            hashFunctions = new List<List<float[]>>();
            hashTables = new List<Dictionary<int, List<int>>>();

            // Initialize tables with NUMBER_OF_HASH_TABLES
            for (int i = 0; i < NUMBER_OF_HASH_TABLES; i++)
            {
                hashTables.Add(new Dictionary<int, List<int>>());
                var tableFunctions = new List<float[]>();
                for (int j = 0; j < NUMBER_OF_HASH_FUNCTIONS; j++)
                {
                    tableFunctions.Add(CreateRandomHashVector());
                }
                hashFunctions.Add(tableFunctions);
            }
            Debug.WriteLine($"Initialized {NUMBER_OF_HASH_TABLES} hash tables");
        }

        private float[] CreateRandomHashVector()
        {
            var random = new Random();
            var vector = new float[MAX_SEQUENCE_LENGTH];
            for (int i = 0; i < MAX_SEQUENCE_LENGTH; i++)
            {
                vector[i] = (float)NextGaussian(random);
            }
            //  Debug.WriteLine($"Created random hash vector of length {vector.Length}");
            return vector;
        }

        private void InitializeLSH2()
        {
            hashFunctions = new List<List<float[]>>();
            hashTables = new List<Dictionary<int, List<int>>>();

            Random random = new Random();
            int embeddingSize = MAX_SEQUENCE_LENGTH; // Using MAX_SEQUENCE_LENGTH as embedding size

            for (int i = 0; i < NUMBER_OF_HASH_TABLES; i++)
            {
                var tableFunctions = new List<float[]>();
                for (int j = 0; j < NUMBER_OF_HASH_FUNCTIONS; j++)
                {
                    var hashVector = Enumerable.Range(0, embeddingSize)
                        .Select(_ => (float)NextGaussian(random))
                        .ToArray();
                    tableFunctions.Add(hashVector);
                }
                hashFunctions.Add(tableFunctions);
                hashTables.Add(new Dictionary<int, List<int>>());
            }
        }

        private void InsertLSH(int id, float[] vector)
        {
            //List<int> hashCodes = ComputeLSH(vector);

            //for (int i = 0; i < NUMBER_OF_HASH_TABLES; i++)
            //{
            //    int hashCode = hashCodes[i];
            //    if (!hashTables[i].ContainsKey(hashCode))
            //    {
            //        hashTables[i][hashCode] = new List<int>();
            //    }
            //    hashTables[i][hashCode].Add(editor_id);
            //}
            List<int> hashCodes = ComputeLSH(vector);
            for (int i = 0; i < NUMBER_OF_HASH_TABLES; i++)
            {
                int hashCode = hashCodes[i];
                if (!hashTables[i].ContainsKey(hashCode))
                {
                    hashTables[i][hashCode] = new List<int>();
                }
                hashTables[i][hashCode].Add(id);
                Debug.WriteLine($"Added ID {id} to table {i}, hashCode {hashCode}");
            }
        }

        private void LogMessage(string message)
        {
            _logBuilder.AppendLine(message);
            Debug.WriteLine(message);
        }






        public async Task<List<FeedbackDatabaseValues>> SearchEmbeddingsAsync(string searchText, int topK, float minimumSimilarity = 0.5f, int searchLevel = 2)
        {
            // Get results from both search methods
            var lshResults = await SearchSimilarEmbeddingsLSHAsync(searchText, topK, minimumSimilarity, searchLevel);
            var memoryResults = await SearchMemoryBasedAsync(searchText, topK);

            // Combine and deduplicate results
            var combinedResults = new Dictionary<string, FeedbackDatabaseValues>();

            foreach (var result in lshResults.Concat(memoryResults))
            {
                if (!combinedResults.ContainsKey(result.RequestID))
                {
                    combinedResults[result.RequestID] = result;
                }
                else
                {
                    // If entry exists, keep the higher similarity score
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





        public async Task<List<FeedbackDatabaseValues>> SearchDatabaseAsync(string searchText, int topK = 5, double maxSimilarity = 0.5, double minSimilarity = 0.2, SimilarityMode mode = SimilarityMode.Normal)
        {
            var connection = new SqliteConnection(_connectionString);
            var similarTexts = new List<FeedbackDatabaseValues>();
            var query = @"SELECT Id, RequestID, UserMessageType, AssistantMessageType, Rating, MetaData,
Request, TextResponse, ToolUseTextResponse, ToolName, ToolContent, Code, Summary, Comment, Errors, RequestStatus,
RequestEmbedding, TextResponseEmbedding, ToolUseTextResponseEmbedding, SummaryEmbedding,
RequestEmbeddingList, TextResponseEmbeddingList, ToolUseTextResponseEmbeddingList, SummaryEmbeddingList";

            //                       RequestID, UserMessageType, AssistantMessageType, Rating, MetaData, Request, TextResponse, ToolUseTextResponse, ToolName, ToolContent, Code, Summary, Comment, Errors, RequestStatus, RequestEmbedding, TextResponseEmbedding, ToolUseTextResponseEmbedding, SummaryEmbedding, RequestEmbeddingList, TextResponseEmbeddingList, ToolUseTextResponseEmbeddingList, SummaryEmbeddingList

            try
            {
                if (connection.State != ConnectionState.Open)
                {
                    await connection.OpenAsync();
                }


                using (var command = new SqliteCommand(query, _connection))
                {
                    command.Parameters.AddWithValue("@SearchText", $"%{searchText}%");


                    using (var reader = await command.ExecuteReaderAsync())
                    {
                        while (await reader.ReadAsync())
                        {
                            var feedbackData = feedbackDatabaseValues(reader);
                            if (!string.IsNullOrEmpty(feedbackData.Request))
                            {
                                double similarity = CalculateTextSimilarity(feedbackData.Request, searchText, mode);

                                if (similarity > minSimilarity)
                                {
                                    feedbackData.Similarity = similarity;
                                    similarTexts.Add(feedbackData);
                                }
                            }
                        }
                    }
                }
            }
            finally
            {
                if (connection.State == ConnectionState.Open)
                {
                    await connection.CloseAsync();
                }
            }

            return similarTexts.OrderByDescending(x => x.Similarity).Take(topK).ToList();
        }

        private async Task<List<FeedbackDatabaseValues>> SearchByTextContentAsync(string text)
        {
            var results = new List<FeedbackDatabaseValues>();
            var connection = new SqliteConnection(_connectionString);

            //const string query = @"
            //SELECT * FROM embeddings
            //WHERE Request LIKE @text
            //   OR TextResponse LIKE @text
            //   OR ToolUseTextResponse LIKE @text";



            const string query = @"
SELECT * FROM embeddings
WHERE Request = @text
   OR Request LIKE '%' || @text || '%'
   OR TextResponse = @text
   OR TextResponse LIKE '%' || @text || '%'
   OR ToolUseTextResponse = @text
   OR ToolUseTextResponse LIKE '%' || @text || '%'";



            try
            {
                await connection.OpenAsync();

                using var command = new SqliteCommand(query, connection);
                //command.Parameters.AddWithValue("@text", $"%{text}%");

                command.Parameters.AddWithValue("@text", $"%{text.Trim()}%");


                using var reader = await command.ExecuteReaderAsync();
                while (await reader.ReadAsync())
                {
                    results.Add(feedbackDatabaseValues(reader));
                }
            }
            finally
            {
                if (connection.State == ConnectionState.Open)
                {
                    await connection.CloseAsync();
                }
            }

            return results;
        }

        private async Task<List<int>> SearchLSHAsync(float[] queryVector, int topK)
        {
            List<int> hashCodes = ComputeLSH(queryVector);
            Debug.WriteLine($"LSH Search - Generated hash codes: {string.Join(", ", hashCodes)}");

            HashSet<int> candidateSet = new HashSet<int>();

            for (int i = 0; i < NUMBER_OF_HASH_TABLES; i++)
            {
                Debug.WriteLine($"Checking table {i}, Has keys: {hashTables[i].Keys.Count}");
                if (hashTables[i].ContainsKey(hashCodes[i]))
                {
                    candidateSet.UnionWith(hashTables[i][hashCodes[i]]);
                    Debug.WriteLine($"Found matches in table {i}: {hashTables[i][hashCodes[i]].Count}");
                }
            }
            Debug.WriteLine($"Total candidates found: {candidateSet.Count}");
            return candidateSet.ToList();
        }

        private async Task<List<int>> SearchLSHAsync2(float[] queryVector, int topK)
        {
            List<int> hashCodes = ComputeLSH(queryVector);
            HashSet<int> candidateSet = new HashSet<int>();

            for (int i = 0; i < NUMBER_OF_HASH_TABLES; i++)
            {
                int hashCode = hashCodes[i];
                if (hashTables[i].ContainsKey(hashCode))
                {
                    candidateSet.UnionWith(hashTables[i][hashCode]);
                }
            }

            var similarities = new List<(int id, double similarity)>();
            foreach (int id in candidateSet)
            {
                float[] candidateVector = await GetVectorByIdAsync(id);  // Properly await the async method
                if (candidateVector != null)
                {
                    double similarity = CalculateSimilarity(queryVector, candidateVector);
                    similarities.Add((id, similarity));
                }
            }

            return similarities.OrderByDescending(x => x.similarity)
                               .Take(topK)
                               .Select(x => x.id)
                               .ToList();
        }

        private async Task<List<int>> SearchLSHAsync3(float[] queryVector, int topK)
        {
            List<int> hashCodes = ComputeLSH(queryVector);
            Debug.WriteLine($"Hash codes: {string.Join(", ", hashCodes)}");

            HashSet<int> candidateSet = new HashSet<int>();

            for (int i = 0; i < NUMBER_OF_HASH_TABLES; i++)
            {
                int hashCode = hashCodes[i];
                Debug.WriteLine($"Checking hash table {i}, hash code {hashCode}");
                if (hashTables[i].ContainsKey(hashCode))
                {
                    Debug.WriteLine($"Found matches in table {i}");
                    candidateSet.UnionWith(hashTables[i][hashCode]);
                }
            }

            Debug.WriteLine($"Found {candidateSet.Count} candidates");
            return candidateSet.OrderByDescending(x => x).Take(topK).ToList();
        }




        private async Task<float[]> GetVectorByIdAsync(int id)
        {
            // Initialize the connection with the specified connection string
            var query = "SELECT RequestEmbedding FROM embeddings WHERE Id = @Id";

            using (var connection = new SqliteConnection(_connectionString))
            {
                try
                {
                    if (connection.State != ConnectionState.Open)
                    {
                        await connection.OpenAsync();
                    }

                    // Prepare the SQL command to retrieve the embedding
                    using (var command = new SqliteCommand(query, connection))
                    {
                        // Parameterize the query to prevent SQL injection
                        command.Parameters.AddWithValue("@Id", id);

                        // Execute the command and retrieve the single result asynchronously
                        var result = await command.ExecuteScalarAsync();

                        // Check if the result is a string and deserialize it into a float array
                        if (result is string resultString)
                        {
                            return JsonConvert.DeserializeObject<float[]>(resultString) ?? Array.Empty<float>();
                        }
                    }
                }
                catch (Exception ex)
                {
                    // Log any exceptions that occur during the database query
                    LogMessage($"Error during database query: {ex.Message}");
                }
                finally
                {
                    // Ensure the database connection is closed if it's still open
                    if (connection.State == ConnectionState.Open)
                    {
                        await connection.CloseAsync();
                    }
                }
            }

            // If no data was found or an error occurred, return an empty array
            return Array.Empty<float>();
        }

        private double CalculateSimilarity(float[] embedding1, float[] embedding2)
        {
            if (embedding1 == null || embedding2 == null || embedding1.Length != embedding2.Length)
            {
                return 0;
            }

            double dotProduct = 0;
            double magnitude1 = 0;
            double magnitude2 = 0;

            for (int i = 0; i < embedding1.Length; i++)
            {
                dotProduct += embedding1[i] * embedding2[i];
                magnitude1 += embedding1[i] * embedding1[i];
                magnitude2 += embedding2[i] * embedding2[i];
            }

            magnitude1 = Math.Sqrt(magnitude1);
            magnitude2 = Math.Sqrt(magnitude2);

            if (magnitude1 == 0 || magnitude2 == 0)
            {
                return 0;
            }

            return dotProduct / (magnitude1 * magnitude2);
        }




        private double CalculateOrderSimilarity(string[] searchWords, string[] docWords)
        {
            List<int> searchWordPositions = new List<int>();
            foreach (string word in searchWords)
            {
                int position = Array.IndexOf(docWords, word);
                if (position != -1)
                {
                    searchWordPositions.Add(position);
                }
            }

            if (searchWordPositions.Count < 2)
            {
                return 0;
            }

            int inversions = 0;
            for (int i = 0; i < searchWordPositions.Count - 1; i++)
            {
                for (int j = i + 1; j < searchWordPositions.Count; j++)
                {
                    if (searchWordPositions[i] > searchWordPositions[j])
                    {
                        inversions++;
                    }
                }
            }

            int maxInversions = (searchWordPositions.Count * (searchWordPositions.Count - 1)) / 2;
            return 1.0 - ((double)inversions / maxInversions);
        }

        private double CalculateWordMatchScore(string[] searchWords, string documentText)
        {
            documentText = documentText.ToLower();
            var docWords = documentText.Split(new[] { ' ', '.', ',', ';', ':', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);

            int matchedWords = searchWords.Count(word => documentText.Contains(word));
            double baseScore = (double)matchedWords / searchWords.Length;

            // Factor 1: Exact phrase match bonus
            double phraseMatchBonus = documentText.Contains(string.Join(" ", searchWords)) ? 0.2 : 0;

            // Factor 2: Word order similarity
            double orderSimilarity = CalculateOrderSimilarity(searchWords, docWords);

            // Factor 3: Document length factor
            double lengthFactor = Math.Min(1.0, (double)docWords.Length / 1000);  // Normalize to 1 for documents with 1000+ words

            // Factor 4: Word density
            double density = (double)matchedWords / docWords.Length;

            // Combine factors
            double score = baseScore + phraseMatchBonus + (orderSimilarity * 0.1) + (lengthFactor * 0.1) + (density * 0.1);

            // Add a small random factor to break ties (0.001 range)
            Random rand = new Random();
            score += rand.NextDouble() * 0.001;

            return Math.Min(score, 1.0);  // Ensure score doesn't exceed 1.0
        }

        private List<int> ComputeLSH(float[] vector)
        {
            List<int> hashCodes = new List<int>();
            Debug.WriteLine($"Computing LSH for vector length: {vector.Length}");

            for (int i = 0; i < NUMBER_OF_HASH_TABLES; i++)
            {
                int hashCode = 0;
                for (int j = 0; j < NUMBER_OF_HASH_FUNCTIONS; j++)
                {
                    float dotProduct = 0;
                    float[] hashVector = hashFunctions[i][j];

                    int minLength = Math.Min(vector.Length, hashVector.Length);
                    for (int k = 0; k < minLength; k++)
                    {
                        dotProduct += vector[k] * hashVector[k];
                    }
                    hashCode = (hashCode << 1) | (dotProduct > 0 ? 1 : 0);
                }
                hashCodes.Add(hashCode);
            }

            return hashCodes;
        }

        private double NextGaussian(Random random)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }

        private string[] PrepareSearchWords(string searchText)
        {
            var stopWords = new HashSet<string> { "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about", "into", "over", "after" };
            return searchText.ToLower()
                             .Split(new[] { ' ', '.', ',', ';', ':', '!', '?' }, StringSplitOptions.RemoveEmptyEntries)
                             .Where(word => !stopWords.Contains(word) && word.Length > 1)
                             .ToArray();
        }

        #endregion LSH

        #region Embedding Methods



        private int model = 0;

        private void PrintModelMetadata(InferenceSession session)
        {
            model++;
            Debug.WriteLine($"ONNX Model {model} Metadata");
            foreach (var input in session.InputMetadata)
            {
                string name = input.Key;
                var metadata = input.Value;

                Debug.WriteLine($"Input Name: {name}");
                Debug.WriteLine($"Data Type: {metadata.ElementType}");
                Debug.WriteLine("Dimensions:");

                if (metadata.Dimensions == null)
                {
                    Debug.WriteLine("  Unknown dimensions.");
                }
                else
                {
                    foreach (var dim in metadata.Dimensions)
                    {
                        Debug.WriteLine($"  {dim.ToString() ?? "Dynamic"}");
                    }
                }

                // Debug.WriteLine(new string('-', 30));
            }

            Debug.WriteLine($"ONNX Model {model} Output Metadata:");
            foreach (var output in session.OutputMetadata)
            {
                string name = output.Key;
                var metadata = output.Value;

                Debug.WriteLine($"Output Name: {name}");
                Debug.WriteLine($"Data Type: {metadata.ElementType}");
                Debug.WriteLine("Dimensions:");

                if (metadata.Dimensions == null)
                {
                    Debug.WriteLine("  Unknown dimensions.");
                }
                else
                {
                    foreach (var dim in metadata.Dimensions)
                    {
                        Debug.WriteLine($"  {dim.ToString() ?? "Dynamic"}");
                    }
                }
            }
            Debug.WriteLine(new string('-', 30));
        }

        private async Task<string> PreprocessTextAsync(string input)
        {
            //  return await Task.Run(() =>
            //  {
            if (string.IsNullOrEmpty(input)) return string.Empty;

            var sb = new StringBuilder(input.Length);
            foreach (char c in input.Normalize(NormalizationForm.FormC))
            {
                if (!char.IsPunctuation(c))
                    sb.Append(c);
            }
            return sb.ToString();
            // });
            await Task.CompletedTask;
        }

        #endregion Embedding Methods

        #region Text Similarity

        private double CalculateTextSimilarity(string searchText, string documentText, SimilarityMode mode = SimilarityMode.Normal)
        {
            // Remove stop words from both search and document text
            searchText = RemoveStopWords(searchText);
            documentText = RemoveStopWords(documentText);

            if (string.IsNullOrWhiteSpace(searchText) || string.IsNullOrWhiteSpace(documentText))
            {
                return 0.0; // No similarity if no valid tokens are left
            }

            // Tokenize the search and document text
            string[] searchTokens = searchText.Split(new[] { ' ', '.', ',', ';', ':', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
            string[] documentTokens = documentText.Split(new[] { ' ', '.', ',', ';', ':', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);

            double similarity = 0.0;

            switch (mode)
            {
                case SimilarityMode.Fast:
                    // Fast mode: Use only Cosine Similarity
                    similarity = CosineSimilarity(searchTokens, documentTokens);
                    break;

                case SimilarityMode.Normal:
                    // Normal mode: Use Cosine Similarity and Jaro-Winkler
                    double cosineSimilarity = CosineSimilarity(searchTokens, documentTokens);
                    double jaroWinklerSimilarity = JaroWinklerDistance(string.Join(" ", searchTokens), string.Join(" ", documentTokens));
                    similarity = (0.5 * cosineSimilarity) + (0.5 * jaroWinklerSimilarity);
                    break;

                case SimilarityMode.Accurate:
                    // Accurate mode: Use Cosine Similarity, Jaro-Winkler, and Levenshtein Distance
                    cosineSimilarity = CosineSimilarity(searchTokens, documentTokens);
                    jaroWinklerSimilarity = JaroWinklerDistance(string.Join(" ", searchTokens), string.Join(" ", documentTokens));
                    double editDistanceSimilarity = 1.0 - (double)LevenshteinDistance(string.Join(" ", searchTokens), string.Join(" ", documentTokens))
                                                    / Math.Max(string.Join(" ", searchTokens).Length, string.Join(" ", documentTokens).Length);
                    similarity = (0.33 * cosineSimilarity) + (0.33 * jaroWinklerSimilarity) + (0.34 * editDistanceSimilarity);
                    break;
            }

            return similarity;
        }

        public enum SimilarityMode
        {
            Fast,
            Normal,
            Accurate
        }

        private double JaroWinklerDistance(string s1, string s2)
        {
            if (s1 == null || s2 == null)
                return 0.0;

            int s1Length = s1.Length;
            int s2Length = s2.Length;

            if (s1Length == 0 || s2Length == 0)
                return 0.0;

            int matchDistance = Math.Max(s1Length, s2Length) / 2 - 1;
            bool[] s1Matches = new bool[s1Length];
            bool[] s2Matches = new bool[s2Length];

            int matches = 0;
            for (int i = 0; i < s1Length; i++)
            {
                int start = Math.Max(0, i - matchDistance);
                int end = Math.Min(i + matchDistance + 1, s2Length);

                for (int j = start; j < end; j++)
                {
                    if (s2Matches[j]) continue;
                    if (s1[i] != s2[j]) continue;
                    s1Matches[i] = true;
                    s2Matches[j] = true;
                    matches++;
                    break;
                }
            }

            if (matches == 0) return 0.0;

            double t = 0;
            int k = 0;
            for (int i = 0; i < s1Length; i++)
            {
                if (!s1Matches[i]) continue;
                while (!s2Matches[k]) k++;
                if (s1[i] != s2[k]) t++;
                k++;
            }

            t /= 2.0;
            double m = matches;

            double jaroSimilarity = (m / s1Length + m / s2Length + (m - t) / m) / 3.0;

            // Jaro-Winkler adjustment
            int prefixLength = 0;
            for (int i = 0; i < Math.Min(4, Math.Min(s1Length, s2Length)); i++)
            {
                if (s1[i] == s2[i])
                    prefixLength++;
                else
                    break;
            }

            double jaroWinklerSimilarity = jaroSimilarity + (0.1 * prefixLength * (1 - jaroSimilarity));

            return jaroWinklerSimilarity;
        }

        private int LevenshteinDistance(string s, string t)
        {
            int n = s.Length;
            int m = t.Length;
            int[,] d = new int[n + 1, m + 1];

            if (n == 0) return m;
            if (m == 0) return n;

            for (int i = 0; i <= n; d[i, 0] = i++) { }
            for (int j = 0; j <= m; d[0, j] = j++) { }

            for (int i = 1; i <= n; i++)
            {
                for (int j = 1; j <= m; j++)
                {
                    int cost = (t[j - 1] == s[i - 1]) ? 0 : 1;
                    d[i, j] = Math.Min(
                        Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                        d[i - 1, j - 1] + cost
                    );
                }
            }
            return d[n, m];
        }

        private double CosineSimilarity(string[] tokens1, string[] tokens2)
        {
            var vector1 = tokens1.GroupBy(t => t).ToDictionary(g => g.Key, g => g.Count());
            var vector2 = tokens2.GroupBy(t => t).ToDictionary(g => g.Key, g => g.Count());

            var allTokens = new HashSet<string>(tokens1.Union(tokens2));
            double dotProduct = 0;
            double magnitude1 = 0;
            double magnitude2 = 0;

            foreach (var token in allTokens)
            {
                int count1 = vector1.ContainsKey(token) ? vector1[token] : 0;
                int count2 = vector2.ContainsKey(token) ? vector2[token] : 0;

                dotProduct += count1 * count2;
                magnitude1 += count1 * count1;
                magnitude2 += count2 * count2;
            }

            magnitude1 = Math.Sqrt(magnitude1);
            magnitude2 = Math.Sqrt(magnitude2);

            if (magnitude1 == 0 || magnitude2 == 0)
            {
                return 0.0;
            }

            return dotProduct / (magnitude1 * magnitude2);
        }


        public Task<(double similarity, double averageTime, double totalTime)> MeasureSimilarityTime(string searchText, string documentText, SimilarityMode mode)
        {
            const int iterations = 10;
            double totalMilliseconds = 0;
            double similarity = 0;

            for (int i = 0; i < iterations; i++)
            {
                Stopwatch stopwatch = Stopwatch.StartNew();
                similarity = CalculateTextSimilarity(searchText, documentText, mode);
                stopwatch.Stop();

                totalMilliseconds += stopwatch.Elapsed.TotalMilliseconds;
            }

            double averageTime = totalMilliseconds / iterations;
            double totalTime = totalMilliseconds;

            return Task.FromResult((similarity, averageTime, totalTime));
        }

        #endregion Text Similarity

        #region Helpers

        private static readonly HashSet<string> stopWords = new HashSet<string>
        {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "it", "as", "that", "this", "these", "those", "are", "was", "were", "be"
        };

        private string RemoveStopWords(string inputText)
        {
            if (string.IsNullOrWhiteSpace(inputText))
            {
                return string.Empty;
            }

            // Split the input text into words
            string[] words = inputText.Split(new[] { ' ', '.', ',', ';', ':', '!', '?', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);

            // Remove stop words
            IEnumerable<string> filteredWords = words.Where(word => !stopWords.Contains(word.ToLower()));

            // Reconstruct the text without stop words
            return string.Join(" ", filteredWords);
        }

        private static string TrimCarriageReturns(string input)
        {
            int startIndex = 0;
            int endIndex = input.Length - 1;

            // Skip leading whitespace including carriage returns
            while (startIndex <= endIndex && char.IsWhiteSpace(input[startIndex]))
            {
                startIndex++;
            }

            // Trim from the end
            while (endIndex >= startIndex && input[endIndex] == '\r')
            {
                endIndex--;
            }

            // Return trimmed string
            return input.Substring(startIndex, endIndex - startIndex + 1);
        }



        private double ScoreContent(string content, double relevance)
        {
            return content.Length * 0.5 + relevance * 0.5;  // Adjust weights as needed
        }



        #endregion Helpers

        private async Task CreateTable()
        {
            var connection = new SqliteConnection(_connectionString);
            var query = @"
CREATE TABLE IF NOT EXISTS embeddings (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    RequestID TEXT,
    UserMessageType TEXT,
    AssistantMessageType TEXT,
    Rating INTEGER,
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
)";
            // RequestID, UserMessageType, AssistantMessageType, Rating, MetaData, Request, TextResponse, ToolUseTextResponse, ToolName, ToolContent, ToolResult, Code, Summary, Comment, Errors, RequestStatus, RequestEmbedding, TextResponseEmbedding, ToolUseTextResponseEmbedding, SummaryEmbedding, RequestEmbeddingList, TextResponseEmbeddingList, ToolUseTextResponseEmbeddingList, SummaryEmbeddingList
            //   RequestID, AssistantMessageType, Rating, MetaData, Request, TextResponse, ToolUseTextResponse, ToolName TEXT, ToolContent, ToolResult, Code, Summary, Comment, Errors, RequestStatus, RequestEmbedding, TextResponseEmbedding, ToolUseTextResponseEmbedding, SummaryEmbedding, RequestEmbeddingList, TextResponseEmbeddingList, ToolUseTextResponseEmbeddingList, SummaryEmbeddingList
            using (var command = new SqliteCommand(query, connection))
            {
                if (connection.State != System.Data.ConnectionState.Open)
                {
                    await connection.OpenAsync();
                }

                try
                {
                    await command.ExecuteNonQueryAsync();
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error adding embeddings: {ex.Message}");
                }
                finally
                {
                    if (connection.State == System.Data.ConnectionState.Open)
                    {
                        await connection.CloseAsync();
                    }
                }
            }
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
)";

            // Insert main data
            await ExecuteCommand(insertQuery, data);

            // Process LSH for single embedding
            if (data.EmbedRequest && data.RequestEmbedding != null)
            {
                InsertLSH(data.Id, data.RequestEmbedding);
            }

            if (data.EmbedTextResponse && data.TextResponseEmbedding != null)
            {
                InsertLSH(data.Id, data.TextResponseEmbedding);
            }

            if (data.EmbedToolUseTextResponse && data.ToolUseTextResponseEmbedding != null)
            {
                InsertLSH(data.Id, data.ToolUseTextResponseEmbedding);
            }

            // Process LSH for embedding list
            if (data.EmbedRequestList && data.RequestEmbeddingList != null)
            {
                foreach (var embedding in data.RequestEmbeddingList)
                {
                    InsertLSH(data.Id, embedding);
                }
            }
            // Process LSH for embedding list
            if (data.EmbedTextResponseList && data.TextResponseEmbeddingList != null)
            {
                foreach (var embedding in data.TextResponseEmbeddingList)
                {
                    InsertLSH(data.Id, embedding);
                }
            }
            // Process LSH for embedding list
            if (data.EmbedToolUseTextResponseList && data.ToolUseTextResponseEmbeddingList != null)
            {
                foreach (var embedding in data.ToolUseTextResponseEmbeddingList)
                {
                    InsertLSH(data.Id, embedding);
                }
            }
        }

        private async Task UpdateDatabaseGood(FeedbackDatabaseValues data)
        {
            const string updateQuery = @"
        UPDATE embeddings
        SET
            UserMessageType = COALESCE(@UserMessageType, UserMessageType),
            AssistantMessageType = COALESCE(@AssistantMessageType, AssistantMessageType),
            Rating = COALESCE(@Rating, Rating),
            MetaData = COALESCE(@MetaData, MetaData),
            Request = COALESCE(@Request, Request),
            TextResponse = CASE
                WHEN @TextResponse IS NOT NULL THEN COALESCE(TextResponse, '') || @TextResponse
                ELSE TextResponse
            END,
            ToolUseTextResponse = CASE
                WHEN @ToolUseTextResponse IS NOT NULL THEN COALESCE(ToolUseTextResponse, '') || @ToolUseTextResponse
                ELSE ToolUseTextResponse
            END,
            ToolName = COALESCE(@ToolName, ToolName),
            ToolContent = CASE
                WHEN @ToolContent IS NOT NULL THEN COALESCE(ToolContent, '') || @ToolContent
                ELSE ToolContent
            END,
            ToolResult = CASE
                WHEN @ToolResult IS NOT NULL THEN COALESCE(ToolResult, '') || @ToolResult
                ELSE ToolResult
            END,
            Code = COALESCE(@Code, Code),
            Summary = COALESCE(@Summary, Summary),
            Comment = CASE
                WHEN @Comment IS NOT NULL THEN COALESCE(Comment, '') || @Comment
                ELSE Comment
            END,
            Errors = CASE
                WHEN @Errors IS NOT NULL THEN COALESCE(Errors, '') || @Errors
                ELSE Errors
            END,
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

            await ExecuteCommand(updateQuery, data);
        }

        private async Task UpdateDatabaseWorking(FeedbackDatabaseValues data)
        {
            const string updateQuery = @"
UPDATE embeddings
SET
    UserMessageType = CASE
       WHEN @UserMessageType IS NOT NULL AND (UserMessageType IS NULL OR UserMessageType = '') THEN @UserMessageType
       ELSE UserMessageType
    END,
    AssistantMessageType = CASE
       WHEN @AssistantMessageType IS NOT NULL AND (AssistantMessageType IS NULL OR AssistantMessageType = '') THEN @AssistantMessageType
       ELSE AssistantMessageType
    END,
    Rating = CASE
       WHEN @Rating IS NOT NULL AND (Rating IS NULL OR Rating = '') THEN @Rating
       ELSE Rating
    END,
    MetaData = CASE
       WHEN @MetaData IS NOT NULL AND (MetaData IS NULL OR MetaData = '') THEN @MetaData
       ELSE MetaData
    END,
    Request = CASE
       WHEN @Request IS NOT NULL AND (Request IS NULL OR Request = '') THEN @Request
       ELSE Request
    END,
    TextResponse = CASE
       WHEN @TextResponse IS NOT NULL THEN IFNULL(TextResponse, '') || @TextResponse
       ELSE TextResponse
    END,
    ToolUseTextResponse = CASE
       WHEN @ToolUseTextResponse IS NOT NULL THEN IFNULL(ToolUseTextResponse, '') || @ToolUseTextResponse
       ELSE ToolUseTextResponse
    END,
    ToolName = CASE
       WHEN @ToolName IS NOT NULL AND (ToolName IS NULL OR ToolName = '') THEN @ToolName
       ELSE ToolName
    END,
    ToolContent = CASE
       WHEN @ToolContent IS NOT NULL THEN IFNULL(ToolContent, '') || @ToolContent
       ELSE ToolContent
    END,
    ToolResult = CASE
       WHEN @ToolResult IS NOT NULL THEN IFNULL(ToolResult, '') || @ToolResult
       ELSE ToolResult
    END,
    Code = CASE
       WHEN @Code IS NOT NULL AND (Code IS NULL OR Code = '') THEN @Code
       ELSE Code
    END,
    Summary = CASE
       WHEN @Summary IS NOT NULL AND (Summary IS NULL OR Summary = '') THEN @Summary
       ELSE Summary
    END,
    Comment = CASE
       WHEN @Comment IS NOT NULL THEN IFNULL(Comment, '') || @Comment
       ELSE Comment
    END,
    Errors = CASE
       WHEN @Errors IS NOT NULL THEN IFNULL(Errors, '') || @Errors
       ELSE Errors
    END,
    RequestStatus = CASE
       WHEN @RequestStatus IS NOT NULL THEN @RequestStatus
       ELSE RequestStatus
   END,
   RequestEmbedding = CASE
       WHEN @RequestEmbedding IS NOT NULL THEN @RequestEmbedding
       ELSE RequestEmbedding
   END,
   TextResponseEmbedding = CASE
       WHEN @TextResponseEmbedding IS NOT NULL THEN @TextResponseEmbedding
       ELSE TextResponseEmbedding
   END,
   ToolUseTextResponseEmbedding = CASE
       WHEN @ToolUseTextResponseEmbedding IS NOT NULL THEN @ToolUseTextResponseEmbedding
       ELSE ToolUseTextResponseEmbedding
   END,
   SummaryEmbedding = CASE
       WHEN @SummaryEmbedding IS NOT NULL THEN @SummaryEmbedding
       ELSE SummaryEmbedding
   END,
   RequestEmbeddingList = CASE
       WHEN @RequestEmbeddingList IS NOT NULL THEN @RequestEmbeddingList
       ELSE RequestEmbeddingList
   END,
   TextResponseEmbeddingList = CASE
       WHEN @TextResponseEmbeddingList IS NOT NULL THEN @TextResponseEmbeddingList
       ELSE TextResponseEmbeddingList
   END,
   ToolUseTextResponseEmbeddingList = CASE
       WHEN @ToolUseTextResponseEmbeddingList IS NOT NULL THEN @ToolUseTextResponseEmbeddingList
       ELSE ToolUseTextResponseEmbeddingList
   END,
   SummaryEmbeddingList = CASE
       WHEN @SummaryEmbeddingList IS NOT NULL THEN @SummaryEmbeddingList
       ELSE SummaryEmbeddingList
   END
WHERE RequestID = @RequestID

";


            await ExecuteCommand(updateQuery, data);
        }

        private async Task UpdateDatabase(FeedbackDatabaseValues data)
        {
            var query = new StringBuilder("UPDATE embeddings SET ");
            var columns = new Dictionary<string, (bool append, bool overwrite)>
           {
               { "UserMessageType", (append: false,overwrite: false) },
               { "AssistantMessageType", (append: false,overwrite:  true) },
               { "Rating", (append: false,overwrite:  true) },
               { "MetaData", (append: false,overwrite:  false) },
               { "Request", (append: false, overwrite: false) },
               { "TextResponse", (append: true,overwrite:  false) },
               { "ToolUseTextResponse", (append: true, overwrite: false) },
               { "ToolName", (append : false, overwrite: false) },
               { "ToolContent", (append : true, overwrite: false) },
               { "ToolResult", (append : true, overwrite: false) },
               { "Code", (append : false, overwrite: false) },
               { "Summary", (append : false, overwrite: false) },
               { "Comment", (append : true, overwrite: false) },
               { "Errors", (append : true, overwrite: false) },
               { "RequestStatus", (append : false, overwrite:  true) },
               { "RequestEmbedding", (append : false, overwrite : true) },
               { "TextResponseEmbedding", (append : false, overwrite : true) },
               { "ToolUseTextResponseEmbedding", (append : false, overwrite : true) },
               { "SummaryEmbedding", (append : false, overwrite : true) },
               { "RequestEmbeddingList", (append : false, overwrite : true) },
               { "TextResponseEmbeddingList", (append : false, overwrite : true) },
               { "ToolUseTextResponseEmbeddingList", (append : false, overwrite : true) },
               { "SummaryEmbeddingList", (append : false, overwrite : true) }
           };

            query.Append(string.Join(",", columns.Select(col =>
                QueryBuilder(col.Key, col.Value.append, col.Value.overwrite))));

            query.Append(" WHERE RequestID = @RequestID");

            await ExecuteCommand(query.ToString(), data);
        }

        private string QueryBuilder(string name, bool append, bool overwrite)
        {
            if (append && overwrite)
                return $"{name} = CASE WHEN @{name} IS NOT NULL THEN @{name} || {name} ELSE {name} END";
            if (append)
                return $"{name} = CASE WHEN @{name} IS NOT NULL THEN IFNULL({name}, '') || @{name} ELSE {name} END";
            if (overwrite)
                return $"{name} = CASE WHEN @{name} IS NOT NULL THEN @{name} ELSE {name} END";

            return $"{name} = CASE WHEN @{name} IS NOT NULL AND ({name} IS NULL OR {name} = '') THEN @{name} ELSE {name} END";
        }

        private async Task<FeedbackDatabaseValues?> GetFeedbackDataFromEmbeddingDatabaseByRequestIDAsync(string requestID)
        {
            if (string.IsNullOrEmpty(requestID))
            {
                throw new ArgumentException("RequestID cannot be null or empty.", nameof(requestID));
            }



            string query = @"SELECT * FROM embeddings WHERE RequestID = @RequestID";

            using (var connection = new SqliteConnection(_connectionString))
            {
                try
                {
                    if (connection.State != ConnectionState.Open)
                    {
                        await connection.OpenAsync();
                    }
                    using (var command = new SqliteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@RequestID", requestID);

                        using (var reader = await command.ExecuteReaderAsync())
                        {
                            if (await reader.ReadAsync())
                            {
                                var feedbackData = feedbackDatabaseValues(reader);
                                return feedbackData;
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error retrieving recent interactions: {ex.Message}");
                }
                finally
                {
                    if (connection.State == ConnectionState.Open)
                    {
                        await connection.CloseAsync();
                    }
                }
            }


            return null; // Return null if no matching record is found
        }

        private async Task<FeedbackDatabaseValues?> GetFeedbackDataDataFromEmbeddingDatabaseById(int id)
        {
            var connection = new SqliteConnection(_connectionString);

            var query = "SELECT * FROM embeddings WHERE Id = @Id";
            try
            {
                using (connection)
                {
                    if (connection.State != ConnectionState.Open)
                    {
                        await connection.OpenAsync();
                    }

                    using (var command = new SqliteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Id", id);

                        using (var reader = await command.ExecuteReaderAsync())
                        {
                            if (await reader.ReadAsync())
                            {
                                var feedbackData = feedbackDatabaseValues(reader);

                                return feedbackData;
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error retrieving recent interactions: {ex.Message}");
            }
            finally
            {
                if (connection.State == System.Data.ConnectionState.Open)
                {
                    await connection.CloseAsync();
                }
            }

            return null;
        }

        private async Task<List<(string request, string textResponse, string toolUseTextResponse, string toolContent, string toolResult, string requestID)>> GetRecentConversationHistoryAsync(int count)
        {
            var recentInteractions = new List<(string request, string textResponse, string toolUseTextResponse, string toolContent, string toolResult, string requestID)>();

            using (var connection = new SqliteConnection(_connectionString))
            using (var command = new SqliteCommand(@"
        SELECT * FROM embeddings
        ORDER BY Id DESC
        LIMIT @Count", connection))
            {
                command.Parameters.AddWithValue("@Count", count);

                try
                {
                    if (connection.State != ConnectionState.Open)
                    {
                        await connection.OpenAsync();
                    }

                    using (var reader = await command.ExecuteReaderAsync())
                    {
                        while (await reader.ReadAsync())
                        {
                            var feedbackData = feedbackDatabaseValues(reader);

                            if (!string.IsNullOrEmpty(feedbackData.TextResponse) || !string.IsNullOrEmpty(feedbackData.ToolUseTextResponse))
                            {
                                recentInteractions.Add((
                                    feedbackData.Request ?? "",         // Use empty string instead of null
                                    feedbackData.TextResponse ?? "",    // Use empty string instead of null
                                    feedbackData.ToolUseTextResponse ?? "", // Use empty string instead of null
                                    feedbackData.ToolContent ?? "", // Use empty string instead of null
                                    feedbackData.ToolResult ?? "", // Use empty string instead of null
                                    feedbackData.RequestID ?? ""        // Use empty string instead of null
                                ));
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error retrieving recent interactions: {ex.Message}");
                }
            }

            recentInteractions.Reverse();
            return recentInteractions;
        }

        private string FormatComment(string comment) => $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}\n\n{comment}\n\n----\n\n";

        private string FormatToolContent(string toolName, string toolContent) => $"{toolName ?? "N/A"}\n{toolContent}\n";

        private string FormatToolResult(string toolName, string toolResult) => $"{toolName ?? "N/A"}\n{toolResult}\n";

        private string FormatTextResponse(string textResponse) => $"{textResponse}\n";

        private string FormatToolUseTextResponse(string toolName, string toolUseTextResponse) => $"{toolName ?? "N/A"}\n{toolUseTextResponse}\n";

        private string FormatToolName(string toolName) => $"{toolName}\n";

        private async Task<bool> RequestIDExistsAsync(string requestId)
        {
            const string checkQuery = "SELECT COUNT(1) FROM embeddings WHERE RequestID = @RequestID";

            using var connection = new SqliteConnection(_connectionString);
            using var checkCommand = new SqliteCommand(checkQuery, connection);

            checkCommand.Parameters.AddWithValue("@RequestID", requestId);

            await connection.OpenAsync();
            var result = Convert.ToInt32(await checkCommand.ExecuteScalarAsync()) > 0;
            await connection.CloseAsync();

            return result;
        }

        private async Task<(bool hasNullEmbeddings, EmbeddingStatus status)> CheckExistingEmbeddings(string requestId)
        {
            Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Checking embeddings for RequestID: {requestId}");

            // First check if the record exists at all
            const string existsQuery = "SELECT COUNT(1) FROM embeddings WHERE RequestID = @RequestID";
            const string embeddingsQuery = @"
        SELECT ToolUseTextResponseEmbedding, ToolUseTextResponseEmbeddingList, TextResponseEmbedding, TextResponseEmbeddingList
        FROM embeddings
        WHERE RequestID = @RequestID";

            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync();

            // First check existence
            using (var existsCommand = new SqliteCommand(existsQuery, connection))
            {
                existsCommand.Parameters.AddWithValue("@RequestID", requestId);
                var count = Convert.ToInt32(await existsCommand.ExecuteScalarAsync());
                if (count == 0)
                {
                    Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} Record does not exist yet for RequestID");
                    return (true, new EmbeddingStatus());
                }
            }

            // If we get here, we know the record exists, now check embeddings
            using (var command = new SqliteCommand(embeddingsQuery, connection))
            {
                command.Parameters.AddWithValue("@RequestID", requestId);
                using var reader = await command.ExecuteReaderAsync();

                if (await reader.ReadAsync())
                {
                    bool singleEmbeddingNull = reader.IsDBNull(0);
                    bool listEmbeddingNull = reader.IsDBNull(1);

                    Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} ToolUseTextResponseEmbedding is null: {singleEmbeddingNull}");
                    Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} ToolUseTextResponseEmbeddingList is null: {listEmbeddingNull}");

                    return (
                        hasNullEmbeddings: singleEmbeddingNull || listEmbeddingNull,
                        status: new EmbeddingStatus
                        {
                            ToolUseTextResponseEmbedding = !singleEmbeddingNull,
                            ToolUseTextResponseEmbeddingList = !listEmbeddingNull
                        }
                    );
                }

                Debug.WriteLine($"{DateTime.Now:HH:mm:ss.fff} No record found in second check");
                return (true, new EmbeddingStatus());
            }
        }

        private async Task GetOrdinals()
        {
            // Dictionary mapping field names to their corresponding index variables
            Dictionary<string, Action<int>> _ordinalSetters = new()
            {
                {"Id", index => idIndex = index},
                {"RequestID", index => requestIdIndex = index},
                {"UserMessageType", index => userMessageTypeIndex = index},
                {"AssistantMessageType", index => assistantMessageTypeIndex = index},
                {"Rating", index => ratingIndex = index},
                {"MetaData", index => metaDataIndex = index},
                {"Request", index => requestIndex = index},
                {"TextResponse", index => textResponseIndex = index},
                {"ToolUseTextResponse", index => toolUseTextResponseIndex = index},
                {"ToolName", index => toolNameIndex = index},
                {"ToolContent", index => toolContentIndex = index},
                {"ToolResult", index => toolResultIndex = index},
                {"Code", index => codeIndex = index},
                {"Summary", index => summaryIndex = index},
                {"Comment", index => commentIndex = index},
                {"Errors", index => errorsIndex = index},
                {"RequestStatus", index => requestStatusIndex = index},
                {"RequestEmbedding", index => requestEmbeddingIndex = index},
                {"RequestEmbeddingList", index => requestEmbeddingListIndex = index},
                {"TextResponseEmbedding", index => textResponseEmbeddingIndex = index},
                {"TextResponseEmbeddingList", index => textResponseEmbeddingListIndex = index},
                {"ToolUseTextResponseEmbedding", index => toolUseTextResponseEmbeddingIndex = index},
                {"ToolUseTextResponseEmbeddingList", index => toolUseTextResponseEmbeddingListIndex = index},
                {"SummaryEmbedding", index => summaryEmbeddingIndex = index},
                {"SummaryEmbeddingList", index => summaryEmbeddingListIndex = index},
                {"Timestamp", index => timestampIndex = index}
            };

            using var connection = new SqliteConnection(_connectionString);
            var query = "SELECT * FROM embeddings LIMIT 1";
            try
            {
                if (connection.State != ConnectionState.Open)
                {
                    await connection.OpenAsync();
                }

                using var command = new SqliteCommand(query, connection);
                using var reader = await command.ExecuteReaderAsync();

                if (await reader.ReadAsync())
                {
                    // Set all ordinals in one loop
                    foreach (var field in _ordinalSetters)
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
                }
            }
            catch (Exception ex)
            {
                LogMessage($"Error during database query: {ex.Message}");
                throw;
            }
            finally
            {
                if (connection.State == ConnectionState.Open)
                    await connection.CloseAsync();
            }
        }

        public async Task<SqliteCommand> ExecuteCommand2(string query, FeedbackDatabaseValues data, SqliteConnection connection)
        {
            using (var command = new SqliteCommand(query, connection))
            {
                command.Parameters.AddWithValue("@RequestID", data.RequestID);
                command.Parameters.AddWithValue("@UserMessageType", data.UserMessageType);
                command.Parameters.AddWithValue("@AssistantMessageType", data.AssistantMessageType);
                command.Parameters.AddWithValue("@Rating", data.Rating);
                command.Parameters.AddWithValue("@MetaData", data.MetaData);
                command.Parameters.AddWithValue("@Request", data.Request);
                command.Parameters.AddWithValue("@TextResponse", data.newTextResponse);
                command.Parameters.AddWithValue("@ToolUseTextResponse", data.newToolUseTextResponse);
                command.Parameters.AddWithValue("@ToolName", data.newToolName);
                command.Parameters.AddWithValue("@ToolContent", data.newToolContent);
                command.Parameters.AddWithValue("@ToolResult", data.newToolResult);
                command.Parameters.AddWithValue("@Code", data.Code);
                command.Parameters.AddWithValue("@Summary", data.Summary);
                command.Parameters.AddWithValue("@Comment", data.newComment);
                command.Parameters.AddWithValue("@Errors", data.Errors);

                command.Parameters.AddWithValue(
                    "@RequestEmbedding", data.RequestEmbedding != null ?
                    JsonConvert.SerializeObject(data.RequestEmbedding) : (object)DBNull.Value);

                command.Parameters.AddWithValue(
                    "@TextResponseEmbedding", data.TextResponseEmbedding != null ?
                    JsonConvert.SerializeObject(data.TextResponseEmbedding) : (object)DBNull.Value);

                command.Parameters.AddWithValue(
                    "@ToolUseTextResponseEmbedding", data.ToolUseTextResponseEmbedding != null ?
                    JsonConvert.SerializeObject(data.ToolUseTextResponseEmbedding) : (object)DBNull.Value);

                command.Parameters.AddWithValue(
                    "@SummaryEmbedding", data.SummaryEmbedding != null ?
                    JsonConvert.SerializeObject(data.SummaryEmbedding) : (object)DBNull.Value);

                // Convert ResponseEmbeddingList to a JSON string
                string? requestEmbeddingListJson = data.RequestEmbeddingList != null ? JsonConvert.SerializeObject(data.RequestEmbeddingList) : null;
                command.Parameters.AddWithValue("@RequestEmbeddingList", requestEmbeddingListJson != null ? (object)requestEmbeddingListJson : DBNull.Value);

                // Convert ResponseEmbeddingList to a JSON string
                string? responseEmbeddingListJson = data.TextResponseEmbeddingList != null ? JsonConvert.SerializeObject(data.TextResponseEmbeddingList) : null;
                command.Parameters.AddWithValue("@ResponseEmbeddingList", responseEmbeddingListJson != null ? (object)responseEmbeddingListJson : DBNull.Value);

                // Convert ResponseEmbeddingList to a JSON string
                string? toolUseResponseEmbeddingListJson = data.ToolUseTextResponseEmbeddingList != null ? JsonConvert.SerializeObject(data.ToolUseTextResponseEmbeddingList) : null;
                command.Parameters.AddWithValue("@ResponseEmbeddingList", responseEmbeddingListJson != null ? (object)responseEmbeddingListJson : DBNull.Value);

                // Convert SummaryEmbeddingList to a JSON string
                string? summaryEmbeddingListJson = data.SummaryEmbeddingList != null ? JsonConvert.SerializeObject(data.SummaryEmbeddingList) : null;
                command.Parameters.AddWithValue("@SummaryEmbeddingList", summaryEmbeddingListJson != null ? (object)summaryEmbeddingListJson : DBNull.Value);

                if (connection.State != ConnectionState.Open)
                {
                    await connection.OpenAsync();
                }

                try
                {
                    await command.ExecuteNonQueryAsync();
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error adding embeddings: {ex.Message}");
                }
                finally
                {
                    if (connection.State == System.Data.ConnectionState.Open)
                    {
                        await connection.CloseAsync();
                    }
                }
            }


            return new SqliteCommand(query, connection);
        }



        private async Task<SqliteCommand> ExecuteCommand(string query, FeedbackDatabaseValues data)
        {
            using var connection = await GetConnectionAsync();
            using var command = new SqliteCommand(query, connection);

            try
            {
                // Helper function to add parameter with null handling
                void AddParameter(string name, object? value) =>
                    command.Parameters.AddWithValue(name, value ?? DBNull.Value);

                // Helper function to handle embedding serialization formatting an array of floats into json
                string? SerializeNullable<T>(T? value) where T : class =>
                    value != null ? JsonConvert.SerializeObject(value) : null;

                // Basic parameters
                var basicParams = new Dictionary<string, object?>
                {
                    {"@RequestID", data.RequestID},
                    {"@UserMessageType", data.UserMessageType},
                    {"@AssistantMessageType", data.AssistantMessageType},
                    {"@Rating", data.Rating},
                    {"@MetaData", data.MetaData},
                    {"@Request", data.Request},
                    {"@TextResponse", data.newTextResponse},
                    {"@ToolUseTextResponse", data.ToolUseTextResponse},
                    {"@ToolName", data.newToolName},
                    {"@ToolContent", data.newToolContent},
                    {"@ToolResult", data.newToolResult},
                    {"@Code", data.Code},
                    {"@Summary", data.Summary},
                    {"@Comment", data.newComment},
                    {"@Errors", data.Errors},
                    {"@RequestStatus", data.RequestStatus}
                };

                // Add basic parameters
                foreach (var param in basicParams)
                {
                    AddParameter(param.Key, param.Value);
                }

                // Embedding parameters
                var embeddingParams = new Dictionary<string, object?>
                {
                    {"@RequestEmbedding", SerializeNullable(data.RequestEmbedding)},
                    {"@TextResponseEmbedding", SerializeNullable(data.TextResponseEmbedding)},
                    {"@ToolUseTextResponseEmbedding", SerializeNullable(data.ToolUseTextResponseEmbedding)},
                    {"@SummaryEmbedding", SerializeNullable(data.SummaryEmbedding)},
                    {"@RequestEmbeddingList", SerializeNullable(data.RequestEmbeddingList)},
                    {"@TextResponseEmbeddingList", SerializeNullable(data.TextResponseEmbeddingList)},
                    {"@ToolUseTextResponseEmbeddingList", SerializeNullable(data.ToolUseTextResponseEmbeddingList)},
                    {"@SummaryEmbeddingList", SerializeNullable(data.SummaryEmbeddingList)}
                };

                // Add embedding parameters
                foreach (var param in embeddingParams)
                {
                    AddParameter(param.Key, param.Value);
                }

                // Set command timeout
                command.CommandTimeout = 30; // 30 seconds timeout

                // Execute the command with retry logic
                int retryCount = 0;
                const int maxRetries = 3;
                const int retryDelayMs = 1000; // 1 second initial delay

                while (true)
                {
                    try
                    {
                        await command.ExecuteNonQueryAsync();
                        break; // success - exit the retry loop
                    }
                    catch (SqliteException ex) when (
                        (ex.SqliteErrorCode == 5 || // SQLITE_BUSY
                         ex.SqliteErrorCode == 6 || // SQLITE_LOCKED
                         ex.SqliteErrorCode == 14) && // SQLITE_CANTOPEN
                        retryCount < maxRetries)
                    {
                        retryCount++;
                        if (retryCount == maxRetries)
                        {
                            throw; // Rethrow if we've exhausted our retries
                        }

                        // Exponential backoff delay
                        int delay = retryDelayMs * (int)Math.Pow(2, retryCount - 1);
                        await Task.Delay(delay);

                        // If connection was closed due to error, reopen it
                        if (connection.State != ConnectionState.Open)
                        {
                            await connection.OpenAsync();
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"Error executing command: {ex.Message}");
                        Debug.WriteLine($"SQL Query: {query}");
                        Debug.WriteLine($"Parameters: {string.Join(", ", command.Parameters.Cast<SqliteParameter>().Select(p => $"{p.ParameterName}={p.Value}"))}");
                        throw; // Rethrow non-retryable exceptions
                    }
                }

                return command;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Fatal error in ExecuteCommand: {ex.Message}");
                Debug.WriteLine($"Stack trace: {ex.StackTrace}");
                throw;
            }
        }

        private FeedbackDatabaseValues feedbackDatabaseValues(DbDataReader reader)
        {
            var feedbackData = new FeedbackDatabaseValues();

            // Helper function to handle string fields with null checks
            string? GetNullableString(int index) =>
                reader.IsDBNull(index) ? null :
                string.IsNullOrWhiteSpace(reader.GetString(index)) ? null :
                reader.GetString(index);

            // Helper function to handle embedding deserialization
            T? DeserializeNullable<T>(int index) where T : class =>
                reader.IsDBNull(index) ? null :
                JsonConvert.DeserializeObject<T>(reader.GetString(index));

            // Basic fields
            feedbackData.Id = reader.GetInt32(idIndex);
            feedbackData.RequestID = reader.IsDBNull(requestIdIndex) ? null : reader.GetString(requestIdIndex);
            feedbackData.Rating = reader.IsDBNull(ratingIndex) ? 0 : reader.GetInt32(ratingIndex);
            feedbackData.Timestamp = reader.IsDBNull(timestampIndex) ? null : reader.GetDateTime(timestampIndex);

            // String fields using helper
            feedbackData.UserMessageType = GetNullableString(userMessageTypeIndex);
            feedbackData.AssistantMessageType = GetNullableString(assistantMessageTypeIndex);
            feedbackData.MetaData = GetNullableString(metaDataIndex);
            feedbackData.Request = GetNullableString(requestIndex);
            feedbackData.TextResponse = GetNullableString(textResponseIndex);
            feedbackData.ToolUseTextResponse = GetNullableString(toolUseTextResponseIndex);
            feedbackData.ToolName = GetNullableString(toolNameIndex);
            feedbackData.ToolContent = GetNullableString(toolContentIndex);
            feedbackData.ToolResult = GetNullableString(toolResultIndex);
            feedbackData.Code = GetNullableString(codeIndex);
            feedbackData.Summary = GetNullableString(summaryIndex);
            feedbackData.Comment = GetNullableString(commentIndex);
            feedbackData.Errors = GetNullableString(errorsIndex);
            feedbackData.RequestStatus = GetNullableString(requestStatusIndex);

            // Embedding fields using helper
            feedbackData.RequestEmbedding = DeserializeNullable<float[]>(requestEmbeddingIndex);
            feedbackData.RequestEmbeddingList = DeserializeNullable<List<float[]>>(requestEmbeddingListIndex);

            feedbackData.TextResponseEmbedding = DeserializeNullable<float[]>(textResponseEmbeddingIndex);
            feedbackData.TextResponseEmbeddingList = DeserializeNullable<List<float[]>>(textResponseEmbeddingListIndex);

            feedbackData.ToolUseTextResponseEmbedding = DeserializeNullable<float[]>(toolUseTextResponseEmbeddingIndex);
            feedbackData.ToolUseTextResponseEmbeddingList = DeserializeNullable<List<float[]>>(toolUseTextResponseEmbeddingListIndex);

            feedbackData.SummaryEmbedding = DeserializeNullable<float[]>(summaryEmbeddingIndex);
            feedbackData.SummaryEmbeddingList = DeserializeNullable<List<float[]>>(summaryEmbeddingListIndex);

            return feedbackData;
        }



        // Optional: Add methods to check queue status
        public int GetQueueLength()
        {
            return _updateQueue.Count;
        }

        public bool IsProcessing()
        {
            return _isProcessing;
        }



        // Improved Dispose pattern
        public async Task DisposeAsync()
        {
            if (_isDisposing) return;

            _isDisposing = true;
            _cancellationTokenSource.Cancel();

            try
            {
                // Wait for processing to complete with timeout
                using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                await Task.WhenAny(_processingTask, Task.Delay(Timeout.Infinite, timeoutCts.Token));
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error during disposal: {ex.Message}");
            }
            finally
            {
                //cache cleanup
                _cleanupTimer?.Dispose();
                _cache.Clear();



                _processingSemaphore.Dispose();
                _queueSemaphore.Dispose();
                _cancellationTokenSource.Dispose();
                _connection?.Dispose();
                _embedder?.Dispose();
            }
        }

        #region Database Maintenance Methods

        /// <summary>
        /// Finds entries with NULL embeddings that need to be backfilled
        /// </summary>
        public async Task<List<(int id, string requestId, bool needsRequestEmbedding, bool needsTextResponseEmbedding, bool needsToolUseEmbedding)>> FindEntriesWithMissingEmbeddingsAsync()
        {
            var results = new List<(int, string, bool, bool, bool)>();

            const string query = @"
                SELECT Id, RequestID, Request, TextResponse, ToolUseTextResponse,
                       RequestEmbedding, TextResponseEmbedding, ToolUseTextResponseEmbedding
                FROM embeddings
                WHERE (Request IS NOT NULL AND Request != '' AND (RequestEmbedding IS NULL OR RequestEmbedding = ''))
                   OR (TextResponse IS NOT NULL AND TextResponse != '' AND (TextResponseEmbedding IS NULL OR TextResponseEmbedding = ''))
                   OR (ToolUseTextResponse IS NOT NULL AND ToolUseTextResponse != '' AND (ToolUseTextResponseEmbedding IS NULL OR ToolUseTextResponseEmbedding = ''))
                ORDER BY Id";

            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync();

            using var command = new SqliteCommand(query, connection);
            using var reader = await command.ExecuteReaderAsync();

            while (await reader.ReadAsync())
            {
                var id = reader.GetInt32(0);
                var requestId = reader.IsDBNull(1) ? "" : reader.GetString(1);

                var hasRequest = !reader.IsDBNull(2) && !string.IsNullOrWhiteSpace(reader.GetString(2));
                var hasTextResponse = !reader.IsDBNull(3) && !string.IsNullOrWhiteSpace(reader.GetString(3));
                var hasToolUse = !reader.IsDBNull(4) && !string.IsNullOrWhiteSpace(reader.GetString(4));

                var hasRequestEmb = !reader.IsDBNull(5) && !string.IsNullOrWhiteSpace(reader.GetString(5));
                var hasTextResponseEmb = !reader.IsDBNull(6) && !string.IsNullOrWhiteSpace(reader.GetString(6));
                var hasToolUseEmb = !reader.IsDBNull(7) && !string.IsNullOrWhiteSpace(reader.GetString(7));

                results.Add((
                    id,
                    requestId,
                    hasRequest && !hasRequestEmb,
                    hasTextResponse && !hasTextResponseEmb,
                    hasToolUse && !hasToolUseEmb
                ));
            }

            Debug.WriteLine($"[FindEntriesWithMissingEmbeddings] Found {results.Count} entries with missing embeddings");
            return results;
        }

        /// <summary>
        /// Backfills embeddings for entries that are missing them
        /// </summary>
        public async Task<(int processed, int succeeded, int failed, List<string> errors)> BackfillMissingEmbeddingsAsync(int batchSize = 10, CancellationToken cancellationToken = default)
        {
            int processed = 0;
            int succeeded = 0;
            int failed = 0;
            var errors = new List<string>();

            try
            {
                var missingEntries = await FindEntriesWithMissingEmbeddingsAsync();
                Debug.WriteLine($"[BackfillMissingEmbeddings] Starting backfill for {missingEntries.Count} entries");

                // Process in batches to avoid overwhelming the system
                for (int i = 0; i < missingEntries.Count; i += batchSize)
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        Debug.WriteLine($"[BackfillMissingEmbeddings] Cancelled after processing {processed} entries");
                        break;
                    }

                    var batch = missingEntries.Skip(i).Take(batchSize).ToList();

                    foreach (var (id, requestId, needsRequest, needsTextResponse, needsToolUse) in batch)
                    {
                        try
                        {
                            processed++;

                            // Fetch the full record
                            var data = await GetRecordByIdAsync(id);
                            if (data == null)
                            {
                                errors.Add($"Record {id} not found");
                                failed++;
                                continue;
                            }

                            // Set embedding flags
                            data.EmbedRequest = needsRequest;
                            data.EmbedRequestList = needsRequest;
                            data.EmbedTextResponse = needsTextResponse;
                            data.EmbedTextResponseList = needsTextResponse;
                            data.EmbedToolUseTextResponse = needsToolUse;
                            data.EmbedToolUseTextResponseList = needsToolUse;

                            // Generate embeddings
                            if (needsRequest || needsTextResponse || needsToolUse)
                            {
                                data = await EmbedResponse(data);
                                await UpdateDataInEmbeddingDatabaseAsync(data);
                                succeeded++;
                                Debug.WriteLine($"[BackfillMissingEmbeddings] Successfully embedded entry {id} ({requestId})");
                            }
                        }
                        catch (Exception ex)
                        {
                            failed++;
                            var error = $"Failed to embed entry {id} ({requestId}): {ex.Message}";
                            errors.Add(error);
                            Debug.WriteLine($"[BackfillMissingEmbeddings] {error}");
                        }
                    }

                    // Small delay between batches
                    if (i + batchSize < missingEntries.Count)
                    {
                        await Task.Delay(100, cancellationToken);
                    }
                }

                Debug.WriteLine($"[BackfillMissingEmbeddings] Complete. Processed: {processed}, Succeeded: {succeeded}, Failed: {failed}");
            }
            catch (Exception ex)
            {
                errors.Add($"Backfill process error: {ex.Message}");
                Debug.WriteLine($"[BackfillMissingEmbeddings] Process error: {ex.Message}");
            }

            return (processed, succeeded, failed, errors);
        }

        /// <summary>
        /// Finds entries where both Request and TextResponse are empty
        /// </summary>
        public async Task<List<(int id, string requestId)>> FindEmptyEntriesAsync()
        {
            var results = new List<(int, string)>();

            const string query = @"
                SELECT Id, RequestID
                FROM embeddings
                WHERE (Request IS NULL OR Request = '')
                  AND (TextResponse IS NULL OR TextResponse = '')
                ORDER BY Id";

            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync();

            using var command = new SqliteCommand(query, connection);
            using var reader = await command.ExecuteReaderAsync();

            while (await reader.ReadAsync())
            {
                var id = reader.GetInt32(0);
                var requestId = reader.IsDBNull(1) ? "" : reader.GetString(1);
                results.Add((id, requestId));
            }

            Debug.WriteLine($"[FindEmptyEntries] Found {results.Count} empty entries");
            return results;
        }

        /// <summary>
        /// Deletes entries where both Request and TextResponse are empty
        /// </summary>
        public async Task<(int deleted, List<string> errors)> DeleteEmptyEntriesAsync()
        {
            int deleted = 0;
            var errors = new List<string>();

            try
            {
                var emptyEntries = await FindEmptyEntriesAsync();
                Debug.WriteLine($"[DeleteEmptyEntries] Starting deletion of {emptyEntries.Count} empty entries");

                const string deleteQuery = "DELETE FROM embeddings WHERE Id = @Id";

                using var connection = new SqliteConnection(_connectionString);
                await connection.OpenAsync();

                using var transaction = connection.BeginTransaction();
                try
                {
                    foreach (var (id, requestId) in emptyEntries)
                    {
                        try
                        {
                            using var command = new SqliteCommand(deleteQuery, connection, transaction);
                            command.Parameters.AddWithValue("@Id", id);
                            await command.ExecuteNonQueryAsync();
                            deleted++;
                            Debug.WriteLine($"[DeleteEmptyEntries] Deleted entry {id} ({requestId})");
                        }
                        catch (Exception ex)
                        {
                            errors.Add($"Failed to delete entry {id} ({requestId}): {ex.Message}");
                        }
                    }

                    transaction.Commit();
                    Debug.WriteLine($"[DeleteEmptyEntries] Successfully deleted {deleted} entries");
                }
                catch (Exception ex)
                {
                    transaction.Rollback();
                    errors.Add($"Transaction failed: {ex.Message}");
                    Debug.WriteLine($"[DeleteEmptyEntries] Transaction error: {ex.Message}");
                }
            }
            catch (Exception ex)
            {
                errors.Add($"Delete process error: {ex.Message}");
                Debug.WriteLine($"[DeleteEmptyEntries] Process error: {ex.Message}");
            }

            return (deleted, errors);
        }

        /// <summary>
        /// Gets database statistics for maintenance reporting
        /// </summary>
        public async Task<DatabaseStatistics> GetDatabaseStatisticsAsync()
        {
            var stats = new DatabaseStatistics();

            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync();

            // Total entries
            using (var cmd = new SqliteCommand("SELECT COUNT(*) FROM embeddings", connection))
            {
                stats.TotalEntries = Convert.ToInt32(await cmd.ExecuteScalarAsync());
            }

            // Entries with missing Request embeddings
            using (var cmd = new SqliteCommand(@"
                SELECT COUNT(*) FROM embeddings
                WHERE Request IS NOT NULL AND Request != ''
                  AND (RequestEmbedding IS NULL OR RequestEmbedding = '')", connection))
            {
                stats.MissingRequestEmbeddings = Convert.ToInt32(await cmd.ExecuteScalarAsync());
            }

            // Entries with missing TextResponse embeddings
            using (var cmd = new SqliteCommand(@"
                SELECT COUNT(*) FROM embeddings
                WHERE TextResponse IS NOT NULL AND TextResponse != ''
                  AND (TextResponseEmbedding IS NULL OR TextResponseEmbedding = '')", connection))
            {
                stats.MissingTextResponseEmbeddings = Convert.ToInt32(await cmd.ExecuteScalarAsync());
            }

            // Entries with missing ToolUseTextResponse embeddings
            using (var cmd = new SqliteCommand(@"
                SELECT COUNT(*) FROM embeddings
                WHERE ToolUseTextResponse IS NOT NULL AND ToolUseTextResponse != ''
                  AND (ToolUseTextResponseEmbedding IS NULL OR ToolUseTextResponseEmbedding = '')", connection))
            {
                stats.MissingToolUseEmbeddings = Convert.ToInt32(await cmd.ExecuteScalarAsync());
            }

            // Empty entries (both Request and TextResponse empty)
            using (var cmd = new SqliteCommand(@"
                SELECT COUNT(*) FROM embeddings
                WHERE (Request IS NULL OR Request = '')
                  AND (TextResponse IS NULL OR TextResponse = '')", connection))
            {
                stats.EmptyEntries = Convert.ToInt32(await cmd.ExecuteScalarAsync());
            }

            // Entries with at least one valid embedding
            using (var cmd = new SqliteCommand(@"
                SELECT COUNT(*) FROM embeddings
                WHERE (RequestEmbedding IS NOT NULL AND RequestEmbedding != '')
                   OR (TextResponseEmbedding IS NOT NULL AND TextResponseEmbedding != '')
                   OR (ToolUseTextResponseEmbedding IS NOT NULL AND ToolUseTextResponseEmbedding != '')", connection))
            {
                stats.EntriesWithEmbeddings = Convert.ToInt32(await cmd.ExecuteScalarAsync());
            }

            Debug.WriteLine($"[DatabaseStatistics] {stats}");
            return stats;
        }

        /// <summary>
        /// Helper method to get a full record by ID
        /// </summary>
        private async Task<FeedbackDatabaseValues?> GetRecordByIdAsync(int id)
        {
            const string query = @"
                SELECT RequestID, Request, TextResponse, ToolUseTextResponse, Summary
                FROM embeddings WHERE Id = @Id";

            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync();

            using var command = new SqliteCommand(query, connection);
            command.Parameters.AddWithValue("@Id", id);

            using var reader = await command.ExecuteReaderAsync();
            if (await reader.ReadAsync())
            {
                return new FeedbackDatabaseValues
                {
                    RequestID = reader.IsDBNull(0) ? null : reader.GetString(0),
                    Request = reader.IsDBNull(1) ? null : reader.GetString(1),
                    TextResponse = reader.IsDBNull(2) ? null : reader.GetString(2),
                    ToolUseTextResponse = reader.IsDBNull(3) ? null : reader.GetString(3),
                    Summary = reader.IsDBNull(4) ? null : reader.GetString(4)
                };
            }

            return null;
        }

        #endregion Database Maintenance Methods

        // Add this method to safely stop processing
        public async Task StopProcessingAsync(TimeSpan? timeout = null)
        {
            _isDisposing = true;
            _cancellationTokenSource.Cancel();

            using var timeoutCts = new CancellationTokenSource(timeout ?? TimeSpan.FromSeconds(30));
            try
            {
                await Task.WhenAny(_processingTask, Task.Delay(Timeout.Infinite, timeoutCts.Token));
            }
            catch (OperationCanceledException)
            {
                Debug.WriteLine("Processing stop timed out");
            }
        }

    }

    /// <summary>
    /// Statistics about the embedding database state
    /// </summary>
    public class DatabaseStatistics
    {
        public int TotalEntries { get; set; }
        public int MissingRequestEmbeddings { get; set; }
        public int MissingTextResponseEmbeddings { get; set; }
        public int MissingToolUseEmbeddings { get; set; }
        public int EmptyEntries { get; set; }
        public int EntriesWithEmbeddings { get; set; }

        public int TotalMissingEmbeddings => MissingRequestEmbeddings + MissingTextResponseEmbeddings + MissingToolUseEmbeddings;
        public double EmbeddingCompletionRate => TotalEntries > 0 ? (double)EntriesWithEmbeddings / TotalEntries * 100 : 0;

        public override string ToString()
        {
            return $@"Database Statistics:
  Total Entries: {TotalEntries}
  Entries with Embeddings: {EntriesWithEmbeddings}
  Empty Entries: {EmptyEntries}
  Missing Embeddings:
    - Request: {MissingRequestEmbeddings}
    - TextResponse: {MissingTextResponseEmbeddings}
    - ToolUse: {MissingToolUseEmbeddings}
    - Total: {TotalMissingEmbeddings}
  Completion Rate: {EmbeddingCompletionRate:F2}%";
        }
    }

}
