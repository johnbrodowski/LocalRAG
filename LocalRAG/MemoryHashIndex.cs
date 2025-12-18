using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using Microsoft.Extensions.Logging;
using System.Numerics;
using Microsoft.Extensions.Caching.Memory;
using System.Diagnostics;

namespace LocalRAG
{
    public class MemoryHashIndex : IDisposable
    {
        // Core storage
        private readonly ConcurrentDictionary<string, byte[]> _vectorHashes;
        private readonly ConcurrentDictionary<string, HashMetadata> _hashMetadata;
        private readonly ConcurrentDictionary<int, HashSet<string>> _hashBuckets;
        private readonly MemoryCache _cache;

        // Configuration
        private readonly MemoryHashOptions _options;
        private readonly ILogger _logger;
        private readonly SemaphoreSlim _processLock;

        public class HashMetadata
        {
            public string Id { get; set; }
            public long Timestamp { get; set; }
            public byte[] Hash { get; set; }
            public HashSet<string> Keywords { get; set; }
            public double[] Metrics { get; set; }
            public Dictionary<string, string> Tags { get; set; }
            public ContentType Type { get; set; }
            public int Version { get; set; }
        }

        public enum ContentType
        {
            Text,
            Code,
            Command,
            Environment,
            Configuration,
            Analytics,
            UserPattern,
            Custom
        }

        public class MemoryHashOptions
        {
            public int MaxItems { get; set; } = 100000;
            public int HashSize { get; set; } = 32;
            public int NumHashFunctions { get; set; } = 8;
            public int NumBuckets { get; set; } = 256;
            public TimeSpan CacheTimeout { get; set; } = TimeSpan.FromHours(24);
            public bool UseBloomFilter { get; set; } = true;
            public double SimilarityThreshold { get; set; } = 0.75;
        }

        public MemoryHashIndex(MemoryHashOptions options = null, ILogger logger = null)
        {
            _options = options ?? new MemoryHashOptions();
            _logger = logger;
            _vectorHashes = new ConcurrentDictionary<string, byte[]>();
            _hashMetadata = new ConcurrentDictionary<string, HashMetadata>();
            _hashBuckets = new ConcurrentDictionary<int, HashSet<string>>();
            _cache = new MemoryCache(new MemoryCacheOptions());
            _processLock = new SemaphoreSlim(1, 1);

            InitializeBuckets();
        }

        private void InitializeBuckets()
        {
            for (int i = 0; i < _options.NumBuckets; i++)
            {
                _hashBuckets[i] = new HashSet<string>();
            }
        }

        public async Task<byte[]> CreateHashAsync(float[] embedding)
        {
            await _processLock.WaitAsync();
            try
            {
                using var sha256 = SHA256.Create();
                var bytes = new byte[embedding.Length * sizeof(float)];
                Buffer.BlockCopy(embedding, 0, bytes, 0, bytes.Length);
                var hash = sha256.ComputeHash(bytes).Take(_options.HashSize).ToArray();
                Debug.WriteLine($"Created hash: {BitConverter.ToString(hash)}"); // Add this
                return hash;
            }
            finally
            {
                _processLock.Release();
            }
        }




        public async Task<bool> AddOrUpdateAsync(string id, float[] vector, Dictionary<string, string> tags = null, ContentType type = ContentType.Text)
        {
            try
            {
                var hash = await CreateHashAsync(vector);
                var metadata = new HashMetadata
                {
                    Id = id,
                    Timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    Hash = hash,
                    Keywords = ExtractKeywords(tags),
                    Tags = tags ?? new Dictionary<string, string>(),
                    Type = type,
                    Version = 1
                };

                _vectorHashes[id] = hash;
                _hashMetadata[id] = metadata;


                var bucketIndices = GetBucketIndices(hash);
                Debug.WriteLine($"Adding vector for ID: {id}"); // Add debug
                Debug.WriteLine($"Bucket indices: {string.Join(", ", bucketIndices)}"); // Add debug

                foreach (var bucketIndex in bucketIndices)
                {
                    _hashBuckets[bucketIndex].Add(id);
                    Debug.WriteLine($"Added ID {id} to bucket {bucketIndex}"); // Add debug
                }

                await PruneIfNeededAsync();
                return true;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, $"Error adding hash for ID {id}");
                return false;
            }
        }

        public async Task<List<SearchResult>> SearchAsync(float[] queryVector, int topK , ContentType? filterType = null)
        {
 
            var queryHash = await CreateHashAsync(queryVector);
            var candidates = new ConcurrentDictionary<string, double>();
            var bucketIndices = GetBucketIndices(queryHash);

            Debug.WriteLine($"Bucket indices: {string.Join(", ", bucketIndices)}");
            Debug.WriteLine($"Total buckets: {_hashBuckets.Count}");

            foreach (var bucketIndex in bucketIndices)
            {
                Debug.WriteLine($"Bucket {bucketIndex} has {_hashBuckets[bucketIndex].Count} items");

                foreach (var id in _hashBuckets[bucketIndex])
                {
                    if (_vectorHashes.TryGetValue(id, out var hash))
                    {
                        var similarity = CalculateHashSimilarity(queryHash, hash);
                        Debug.WriteLine($"ID: {id}, Similarity: {similarity}, Threshold: {_options.SimilarityThreshold}");
                        if (similarity >= _options.SimilarityThreshold)
                        {
                            Debug.WriteLine($"Adding to candidates"); // Add this
                            candidates.TryAdd(id, similarity);
                        }
                    }
                }
            }

            Debug.WriteLine($"Total candidates: {candidates.Count}");

            var results = candidates
                .OrderByDescending(x => x.Value)
                .Take(topK)
                .Select(x => {
                    Debug.WriteLine($"Creating SearchResult for ID: {x.Key}");
                    var metadata = _hashMetadata[x.Key];
                    Debug.WriteLine($"Metadata text: {metadata.Tags["text"]}");
                    return new SearchResult
                    {
                        Id = x.Key,
                        Score = x.Value,
                        Metadata = metadata
                    };
                })
                .ToList();

            Debug.WriteLine($"Final results count: {results.Count}");
            return results;

            //return candidates
            //    .OrderByDescending(x => x.Value)
            //    .Take(topK)
            //    .Select(x => new SearchResult
            //    {
            //        editor_id = x.Key,
            //        Score = x.Value,
            //        Metadata = _hashMetadata[x.Key]
            //    })
            //    .ToList();
        }

        private HashSet<string> ExtractKeywords(Dictionary<string, string> tags)
        {
            if (tags == null) return new HashSet<string>();
            return new HashSet<string>(tags.Values.SelectMany(v => v.Split(' ', StringSplitOptions.RemoveEmptyEntries)));
        }

        private int[] GetBucketIndices(byte[] hash)
        {
            var indices = new int[_options.NumHashFunctions];
            for (int i = 0; i < _options.NumHashFunctions; i++)
            {
                // Use a consistent window of the hash for each function
                int startIndex = (i * hash.Length) / _options.NumHashFunctions;
                var portion = hash.Skip(startIndex).Take(4).ToArray();
                indices[i] = Math.Abs(BitConverter.ToInt32(portion, 0) % _options.NumBuckets);
                Debug.WriteLine($"Hash function {i}: Using bytes {startIndex}-{startIndex + 3} -> bucket {indices[i]}");
            }
            return indices;
        }

        private double CalculateHashSimilarity(byte[] hash1, byte[] hash2)
        {
            int matching = 0;
            for (int i = 0; i < hash1.Length; i++)
            {
                matching += BitOperations.PopCount((uint)(hash1[i] ^ (byte.MaxValue ^ hash2[i])));
            }
            return (double)matching / (hash1.Length * 8);
        }

        private async Task PruneIfNeededAsync()
        {
            if (_vectorHashes.Count > _options.MaxItems)
            {
                await _processLock.WaitAsync();
                try
                {
                    var itemsToRemove = _hashMetadata
                        .OrderBy(x => x.Value.Timestamp)
                        .Take(_vectorHashes.Count - (_options.MaxItems * 9 / 10))
                        .Select(x => x.Key)
                        .ToList();

                    foreach (var id in itemsToRemove)
                    {
                        _vectorHashes.TryRemove(id, out _);
                        _hashMetadata.TryRemove(id, out var metadata);

                        if (metadata != null)
                        {
                            var bucketIndices = GetBucketIndices(metadata.Hash);
                            foreach (var bucketIndex in bucketIndices)
                            {
                                _hashBuckets[bucketIndex].Remove(id);
                            }
                        }
                    }
                }
                finally
                {
                    _processLock.Release();
                }
            }
        }

        public class SearchResult
        {
            public string Id { get; set; }
            public double Score { get; set; }
            public HashMetadata Metadata { get; set; }
        }

        public void Dispose()
        {
            _cache.Dispose();
            _processLock.Dispose();
        }
    }
}
