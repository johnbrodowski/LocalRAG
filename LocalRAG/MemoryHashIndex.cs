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
        // Core storage - store actual vectors for cosine similarity
        private readonly ConcurrentDictionary<string, float[]> _vectors;
        private readonly ConcurrentDictionary<string, HashMetadata> _hashMetadata;
        private readonly ConcurrentDictionary<int, HashSet<string>> _hashBuckets;
        private readonly MemoryCache _cache;

        // LSH: Random hyperplanes for locality-sensitive hashing
        private readonly float[][] _hyperplanes;

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

        // Default embedding dimension for BERT base model
        private const int DefaultEmbeddingDimension = 768;

        public MemoryHashIndex(MemoryHashOptions options = null, ILogger logger = null)
        {
            _options = options ?? new MemoryHashOptions();
            _logger = logger;
            _vectors = new ConcurrentDictionary<string, float[]>();
            _hashMetadata = new ConcurrentDictionary<string, HashMetadata>();
            _hashBuckets = new ConcurrentDictionary<int, HashSet<string>>();
            _cache = new MemoryCache(new MemoryCacheOptions());
            _processLock = new SemaphoreSlim(1, 1);

            // Initialize random hyperplanes for LSH
            // Each hyperplane is used to partition the vector space
            _hyperplanes = InitializeHyperplanes(_options.NumHashFunctions, DefaultEmbeddingDimension);

            InitializeBuckets();
        }

        /// <summary>
        /// Initializes random hyperplanes for LSH (Locality Sensitive Hashing).
        /// For cosine similarity, we use random hyperplane method where vectors
        /// on the same side of a hyperplane get the same bit.
        /// </summary>
        private float[][] InitializeHyperplanes(int numHyperplanes, int dimension)
        {
            var random = new Random(42); // Fixed seed for reproducibility
            var hyperplanes = new float[numHyperplanes][];

            for (int i = 0; i < numHyperplanes; i++)
            {
                hyperplanes[i] = new float[dimension];
                for (int j = 0; j < dimension; j++)
                {
                    // Generate random Gaussian values using Box-Muller transform
                    double u1 = 1.0 - random.NextDouble();
                    double u2 = 1.0 - random.NextDouble();
                    hyperplanes[i][j] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
                }
            }

            return hyperplanes;
        }

        private void InitializeBuckets()
        {
            for (int i = 0; i < _options.NumBuckets; i++)
            {
                _hashBuckets[i] = new HashSet<string>();
            }
        }

        /// <summary>
        /// Computes LSH hash using random hyperplane method.
        /// Each bit represents which side of a hyperplane the vector falls on.
        /// Similar vectors will have similar hash codes (high probability of same bits).
        /// </summary>
        public int ComputeLSHHash(float[] embedding)
        {
            int hash = 0;
            int minLen = Math.Min(embedding.Length, DefaultEmbeddingDimension);

            for (int i = 0; i < _hyperplanes.Length; i++)
            {
                // Compute dot product with hyperplane
                double dotProduct = 0;
                for (int j = 0; j < minLen; j++)
                {
                    dotProduct += embedding[j] * _hyperplanes[i][j];
                }

                // Set bit based on sign of dot product
                if (dotProduct >= 0)
                {
                    hash |= (1 << i);
                }
            }

            Debug.WriteLine($"Computed LSH hash: {hash}");
            return hash;
        }




        public async Task<bool> AddOrUpdateAsync(string id, float[] vector, Dictionary<string, string> tags = null, ContentType type = ContentType.Text)
        {
            try
            {
                // Store the actual vector for cosine similarity calculation
                _vectors[id] = vector;

                var lshHash = ComputeLSHHash(vector);
                var metadata = new HashMetadata
                {
                    Id = id,
                    Timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    Hash = BitConverter.GetBytes(lshHash), // Store LSH hash for reference
                    Keywords = ExtractKeywords(tags),
                    Tags = tags ?? new Dictionary<string, string>(),
                    Type = type,
                    Version = 1
                };

                _hashMetadata[id] = metadata;

                // Use LSH hash to determine bucket
                int bucketIndex = Math.Abs(lshHash % _options.NumBuckets);
                Debug.WriteLine($"Adding vector for ID: {id}");
                Debug.WriteLine($"LSH hash: {lshHash}, bucket: {bucketIndex}");

                _hashBuckets[bucketIndex].Add(id);

                await PruneIfNeededAsync();
                return true;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, $"Error adding hash for ID {id}");
                return false;
            }
        }

        public async Task<List<SearchResult>> SearchAsync(float[] queryVector, int topK, ContentType? filterType = null)
        {
            var candidates = new ConcurrentDictionary<string, double>();

            // Use LSH to find candidate bucket
            var queryLSHHash = ComputeLSHHash(queryVector);
            int primaryBucket = Math.Abs(queryLSHHash % _options.NumBuckets);

            // Also check neighboring buckets (LSH can have collisions)
            var bucketsToSearch = new HashSet<int> { primaryBucket };

            // Add nearby buckets for better recall
            bucketsToSearch.Add((primaryBucket + 1) % _options.NumBuckets);
            bucketsToSearch.Add((primaryBucket - 1 + _options.NumBuckets) % _options.NumBuckets);

            Debug.WriteLine($"Query LSH hash: {queryLSHHash}, searching buckets: {string.Join(", ", bucketsToSearch)}");
            Debug.WriteLine($"Total buckets: {_hashBuckets.Count}");

            foreach (var bucketIndex in bucketsToSearch)
            {
                if (!_hashBuckets.TryGetValue(bucketIndex, out var bucket))
                    continue;

                Debug.WriteLine($"Bucket {bucketIndex} has {bucket.Count} items");

                foreach (var id in bucket)
                {
                    // Calculate ACTUAL cosine similarity using stored vectors
                    if (_vectors.TryGetValue(id, out var storedVector))
                    {
                        var similarity = CalculateCosineSimilarity(queryVector, storedVector);
                        Debug.WriteLine($"ID: {id}, Cosine Similarity: {similarity:F4}, Threshold: {_options.SimilarityThreshold}");

                        if (similarity >= _options.SimilarityThreshold)
                        {
                            Debug.WriteLine($"Adding to candidates");
                            candidates.TryAdd(id, similarity);
                        }
                    }
                }
            }

            Debug.WriteLine($"Total candidates: {candidates.Count}");

            var results = candidates
                .OrderByDescending(x => x.Value)
                .Take(topK)
                .Select(x =>
                {
                    Debug.WriteLine($"Creating SearchResult for ID: {x.Key}");
                    var metadata = _hashMetadata[x.Key];
                    if (metadata.Tags.TryGetValue("text", out var text))
                    {
                        Debug.WriteLine($"Metadata text: {text}");
                    }
                    return new SearchResult
                    {
                        Id = x.Key,
                        Score = x.Value,
                        Metadata = metadata
                    };
                })
                .ToList();

            Debug.WriteLine($"Final results count: {results.Count}");
            return await Task.FromResult(results);
        }

        /// <summary>
        /// Calculates cosine similarity between two vectors.
        /// Returns value between -1 and 1, where 1 means identical direction.
        /// </summary>
        private double CalculateCosineSimilarity(float[] vectorA, float[] vectorB)
        {
            if (vectorA == null || vectorB == null)
                return 0;

            int minLen = Math.Min(vectorA.Length, vectorB.Length);
            if (minLen == 0)
                return 0;

            double dotProduct = 0;
            double magnitudeA = 0;
            double magnitudeB = 0;

            for (int i = 0; i < minLen; i++)
            {
                dotProduct += vectorA[i] * vectorB[i];
                magnitudeA += vectorA[i] * vectorA[i];
                magnitudeB += vectorB[i] * vectorB[i];
            }

            magnitudeA = Math.Sqrt(magnitudeA);
            magnitudeB = Math.Sqrt(magnitudeB);

            if (magnitudeA == 0 || magnitudeB == 0)
                return 0;

            return dotProduct / (magnitudeA * magnitudeB);
        }

        private HashSet<string> ExtractKeywords(Dictionary<string, string> tags)
        {
            if (tags == null) return new HashSet<string>();
            return new HashSet<string>(tags.Values.SelectMany(v => v.Split(' ', StringSplitOptions.RemoveEmptyEntries)));
        }

        // Old hash-based methods removed - using proper LSH and cosine similarity now

        private async Task PruneIfNeededAsync()
        {
            if (_vectors.Count > _options.MaxItems)
            {
                await _processLock.WaitAsync();
                try
                {
                    var itemsToRemove = _hashMetadata
                        .OrderBy(x => x.Value.Timestamp)
                        .Take(_vectors.Count - (_options.MaxItems * 9 / 10))
                        .Select(x => x.Key)
                        .ToList();

                    foreach (var id in itemsToRemove)
                    {
                        _vectors.TryRemove(id, out var removedVector);
                        _hashMetadata.TryRemove(id, out var metadata);

                        // Remove from bucket using the vector's LSH hash
                        if (removedVector != null)
                        {
                            var lshHash = ComputeLSHHash(removedVector);
                            int bucketIndex = Math.Abs(lshHash % _options.NumBuckets);
                            if (_hashBuckets.TryGetValue(bucketIndex, out var bucket))
                            {
                                bucket.Remove(id);
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
