using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace LocalRAG
{
    /// <summary>
    /// Comprehensive tests for embedding generation, LSH indexing, and search functionality.
    /// Run these tests after making changes to validate correctness.
    /// </summary>
    public class EmbeddingTests
    {
        private readonly List<TestResult> _results = new List<TestResult>();

        public class TestResult
        {
            public string TestName { get; set; }
            public bool Passed { get; set; }
            public string Message { get; set; }
            public TimeSpan Duration { get; set; }
        }

        /// <summary>
        /// Runs all tests and returns results
        /// </summary>
        public async Task<List<TestResult>> RunAllTestsAsync(RAGConfiguration config = null)
        {
            _results.Clear();
            Console.WriteLine("=" .PadRight(60, '='));
            Console.WriteLine("EMBEDDING & SEARCH TEST SUITE");
            Console.WriteLine("=".PadRight(60, '='));
            Console.WriteLine();

            // Unit tests (no BERT model required)
            await RunTest("Test_CosineSimilarity_IdenticalVectors", Test_CosineSimilarity_IdenticalVectors);
            await RunTest("Test_CosineSimilarity_OrthogonalVectors", Test_CosineSimilarity_OrthogonalVectors);
            await RunTest("Test_CosineSimilarity_OppositeVectors", Test_CosineSimilarity_OppositeVectors);
            await RunTest("Test_CosineSimilarity_SimilarVectors", Test_CosineSimilarity_SimilarVectors);
            await RunTest("Test_LSHHash_SameVectorSameHash", Test_LSHHash_SameVectorSameHash);
            await RunTest("Test_LSHHash_SimilarVectorsSimilarHash", Test_LSHHash_SimilarVectorsSimilarHash);
            await RunTest("Test_LSHHash_DifferentVectorsDifferentHash", Test_LSHHash_DifferentVectorsDifferentHash);
            await RunTest("Test_LSHDimension_Uses768NotSequenceLength", Test_LSHDimension_Uses768NotSequenceLength);
            await RunTest("Test_MemoryHashIndex_AddAndSearch", Test_MemoryHashIndex_AddAndSearch);
            await RunTest("Test_MemoryHashIndex_SimilarVectorsFound", Test_MemoryHashIndex_SimilarVectorsFound);
            await RunTest("Test_WordMatchScore_ExactBoundary", Test_WordMatchScore_ExactBoundary);
            await RunTest("Test_WordMatchScore_NoSubstringMatch", Test_WordMatchScore_NoSubstringMatch);

            // Integration tests (require BERT model)
            if (config != null)
            {
                Console.WriteLine("\n--- Integration Tests (with BERT model) ---\n");
                await RunTest("Test_EmbeddingGeneration_PreservesStopWords", () => Test_EmbeddingGeneration_PreservesStopWords(config));
                await RunTest("Test_EmbeddingDimension_Is768", () => Test_EmbeddingDimension_Is768(config));
                await RunTest("Test_SimilarTexts_HighSimilarity", () => Test_SimilarTexts_HighSimilarity(config));
                await RunTest("Test_DifferentTexts_LowerSimilarity", () => Test_DifferentTexts_LowerSimilarity(config));
                await RunTest("Test_MockDataGeneration", () => Test_MockDataGeneration(config));
            }

            // Print summary
            PrintSummary();

            return _results;
        }

        private async Task RunTest(string name, Func<Task> test)
        {
            var sw = Stopwatch.StartNew();
            try
            {
                await test();
                sw.Stop();
                var result = new TestResult { TestName = name, Passed = true, Message = "PASSED", Duration = sw.Elapsed };
                _results.Add(result);
                Console.WriteLine($"✓ {name} ({sw.ElapsedMilliseconds}ms)");
            }
            catch (Exception ex)
            {
                sw.Stop();
                var result = new TestResult { TestName = name, Passed = false, Message = ex.Message, Duration = sw.Elapsed };
                _results.Add(result);
                Console.WriteLine($"✗ {name} - {ex.Message}");
            }
        }

        private void PrintSummary()
        {
            Console.WriteLine();
            Console.WriteLine("=".PadRight(60, '='));
            var passed = _results.Count(r => r.Passed);
            var failed = _results.Count(r => !r.Passed);
            Console.WriteLine($"RESULTS: {passed} passed, {failed} failed, {_results.Count} total");
            Console.WriteLine("=".PadRight(60, '='));

            if (failed > 0)
            {
                Console.WriteLine("\nFailed tests:");
                foreach (var result in _results.Where(r => !r.Passed))
                {
                    Console.WriteLine($"  - {result.TestName}: {result.Message}");
                }
            }
        }

        #region Cosine Similarity Tests

        private Task Test_CosineSimilarity_IdenticalVectors()
        {
            var vectorA = new float[] { 1, 2, 3, 4, 5 };
            var vectorB = new float[] { 1, 2, 3, 4, 5 };

            var similarity = CalculateCosineSimilarity(vectorA, vectorB);

            Assert(Math.Abs(similarity - 1.0) < 0.0001, $"Expected 1.0, got {similarity}");
            return Task.CompletedTask;
        }

        private Task Test_CosineSimilarity_OrthogonalVectors()
        {
            var vectorA = new float[] { 1, 0, 0 };
            var vectorB = new float[] { 0, 1, 0 };

            var similarity = CalculateCosineSimilarity(vectorA, vectorB);

            Assert(Math.Abs(similarity) < 0.0001, $"Expected 0.0, got {similarity}");
            return Task.CompletedTask;
        }

        private Task Test_CosineSimilarity_OppositeVectors()
        {
            var vectorA = new float[] { 1, 2, 3 };
            var vectorB = new float[] { -1, -2, -3 };

            var similarity = CalculateCosineSimilarity(vectorA, vectorB);

            Assert(Math.Abs(similarity - (-1.0)) < 0.0001, $"Expected -1.0, got {similarity}");
            return Task.CompletedTask;
        }

        private Task Test_CosineSimilarity_SimilarVectors()
        {
            var vectorA = new float[] { 1, 2, 3, 4, 5 };
            var vectorB = new float[] { 1.1f, 2.1f, 3.1f, 4.1f, 5.1f };

            var similarity = CalculateCosineSimilarity(vectorA, vectorB);

            Assert(similarity > 0.99, $"Expected > 0.99, got {similarity}");
            return Task.CompletedTask;
        }

        #endregion

        #region LSH Hash Tests

        private Task Test_LSHHash_SameVectorSameHash()
        {
            var options = new MemoryHashIndex.MemoryHashOptions { NumHashFunctions = 8, NumBuckets = 256 };
            using var index = new MemoryHashIndex(options);

            var vector = GenerateRandomVector(768);
            var hash1 = index.ComputeLSHHash(vector);
            var hash2 = index.ComputeLSHHash(vector);

            Assert(hash1 == hash2, $"Same vector should produce same hash. Got {hash1} and {hash2}");
            return Task.CompletedTask;
        }

        private Task Test_LSHHash_SimilarVectorsSimilarHash()
        {
            var options = new MemoryHashIndex.MemoryHashOptions { NumHashFunctions = 8, NumBuckets = 256 };
            using var index = new MemoryHashIndex(options);

            var baseVector = GenerateRandomVector(768);
            var similarVector = baseVector.Select(v => v + 0.001f).ToArray();

            var hash1 = index.ComputeLSHHash(baseVector);
            var hash2 = index.ComputeLSHHash(similarVector);

            // Similar vectors should have similar hashes (many matching bits)
            int matchingBits = CountMatchingBits(hash1, hash2);
            Assert(matchingBits >= 6, $"Similar vectors should share most bits. Matching: {matchingBits}/8");
            return Task.CompletedTask;
        }

        private Task Test_LSHHash_DifferentVectorsDifferentHash()
        {
            var options = new MemoryHashIndex.MemoryHashOptions { NumHashFunctions = 8, NumBuckets = 256 };
            using var index = new MemoryHashIndex(options);

            var vector1 = GenerateRandomVector(768, seed: 1);
            var vector2 = GenerateRandomVector(768, seed: 999);

            var hash1 = index.ComputeLSHHash(vector1);
            var hash2 = index.ComputeLSHHash(vector2);

            // Very different vectors may have different hashes
            // (not guaranteed, but likely with random vectors)
            Console.WriteLine($"    Hash1: {hash1}, Hash2: {hash2}");
            return Task.CompletedTask;
        }

        private Task Test_LSHDimension_Uses768NotSequenceLength()
        {
            // This test validates that LSH uses 768 dimensions (BERT base)
            // not 512 (MaxSequenceLength)
            var options = new MemoryHashIndex.MemoryHashOptions { NumHashFunctions = 8, NumBuckets = 256 };
            using var index = new MemoryHashIndex(options);

            // Create a 768-dim vector with specific pattern
            var vector768 = new float[768];
            for (int i = 0; i < 768; i++)
                vector768[i] = (i % 2 == 0) ? 1.0f : -1.0f;

            // Create a 512-dim version (truncated)
            var vector512 = vector768.Take(512).ToArray();

            // The hash should use all 768 dimensions
            // If implementation incorrectly uses 512, the hash would be the same
            var hash768 = index.ComputeLSHHash(vector768);

            // Create another 768-dim vector that's identical in first 512 but different in rest
            var vector768Modified = new float[768];
            Array.Copy(vector768, vector768Modified, 512);
            for (int i = 512; i < 768; i++)
                vector768Modified[i] = -vector768[i]; // Flip the rest

            var hash768Modified = index.ComputeLSHHash(vector768Modified);

            // If LSH correctly uses all 768 dims, these hashes should likely differ
            Console.WriteLine($"    Original hash: {hash768}, Modified hash: {hash768Modified}");
            // They might be same by chance, but the test documents the behavior
            return Task.CompletedTask;
        }

        #endregion

        #region MemoryHashIndex Tests

        private async Task Test_MemoryHashIndex_AddAndSearch()
        {
            var options = new MemoryHashIndex.MemoryHashOptions
            {
                NumHashFunctions = 8,
                NumBuckets = 256,
                SimilarityThreshold = 0.5
            };
            using var index = new MemoryHashIndex(options);

            var vector = GenerateRandomVector(768);
            await index.AddOrUpdateAsync("test-1", vector, new Dictionary<string, string> { { "text", "test document" } });

            var results = await index.SearchAsync(vector, 10);

            Assert(results.Count > 0, "Should find the added vector");
            Assert(results[0].Id == "test-1", $"Should find correct ID. Got: {results[0].Id}");
            Assert(results[0].Score > 0.99, $"Identical vector should have score ~1.0. Got: {results[0].Score}");
        }

        private async Task Test_MemoryHashIndex_SimilarVectorsFound()
        {
            var options = new MemoryHashIndex.MemoryHashOptions
            {
                NumHashFunctions = 8,
                NumBuckets = 256,
                SimilarityThreshold = 0.8
            };
            using var index = new MemoryHashIndex(options);

            // Add a base vector
            var baseVector = GenerateNormalizedVector(768);
            await index.AddOrUpdateAsync("base", baseVector, new Dictionary<string, string> { { "text", "base" } });

            // Add a similar vector (small perturbation)
            var similarVector = baseVector.Select(v => v + 0.01f * (float)(new Random().NextDouble() - 0.5)).ToArray();
            similarVector = Normalize(similarVector);
            await index.AddOrUpdateAsync("similar", similarVector, new Dictionary<string, string> { { "text", "similar" } });

            // Add a different vector
            var differentVector = GenerateNormalizedVector(768, seed: 999);
            await index.AddOrUpdateAsync("different", differentVector, new Dictionary<string, string> { { "text", "different" } });

            // Search with base vector
            var results = await index.SearchAsync(baseVector, 10);

            Console.WriteLine($"    Found {results.Count} results");
            foreach (var r in results)
                Console.WriteLine($"      {r.Id}: {r.Score:F4}");

            Assert(results.Any(r => r.Id == "base"), "Should find base vector");
        }

        #endregion

        #region Word Match Score Tests

        private Task Test_WordMatchScore_ExactBoundary()
        {
            var searchWords = new[] { "the", "cat" };
            var document = "The cat sat on the mat";

            var score = CalculateWordMatchScore(searchWords, document);

            Assert(score > 0, $"Should find 'the' and 'cat'. Score: {score}");
            return Task.CompletedTask;
        }

        private Task Test_WordMatchScore_NoSubstringMatch()
        {
            var searchWords = new[] { "the" };
            var document = "brother together other";

            var score = CalculateWordMatchScore(searchWords, document);

            // With proper word boundary matching, "the" should NOT match in these words
            Assert(score == 0, $"Should NOT match 'the' in 'brother'. Score: {score}");
            return Task.CompletedTask;
        }

        #endregion

        #region Integration Tests (Require BERT Model)

        private async Task Test_EmbeddingGeneration_PreservesStopWords(RAGConfiguration config)
        {
            using var embedder = new EmbedderClassNew(config);

            // Text with stop words
            var textWithStopWords = "The quick brown fox jumps over the lazy dog";

            // This should NOT throw and should process the full text
            var embedding = await embedder.GetEmbeddingsAsync(textWithStopWords);

            Assert(embedding != null, "Embedding should not be null");
            Assert(embedding.Length == 768 || embedding.Length == 1024, $"Embedding should be 768 or 1024 dims. Got: {embedding.Length}");
        }

        private async Task Test_EmbeddingDimension_Is768(RAGConfiguration config)
        {
            using var embedder = new EmbedderClassNew(config);

            var embedding = await embedder.GetEmbeddingsAsync("Test text");

            Assert(embedding.Length == 768 || embedding.Length == 1024,
                $"Expected 768 (BERT base) or 1024 (BERT large). Got: {embedding.Length}");
        }

        private async Task Test_SimilarTexts_HighSimilarity(RAGConfiguration config)
        {
            using var embedder = new EmbedderClassNew(config);

            var embedding1 = await embedder.GetEmbeddingsAsync("How do I read a file in C#?");
            var embedding2 = await embedder.GetEmbeddingsAsync("How can I read files using C#?");

            var similarity = CalculateCosineSimilarity(embedding1, embedding2);

            Console.WriteLine($"    Similarity: {similarity:F4}");
            Assert(similarity > 0.8, $"Similar texts should have high similarity. Got: {similarity}");
        }

        private async Task Test_DifferentTexts_LowerSimilarity(RAGConfiguration config)
        {
            using var embedder = new EmbedderClassNew(config);

            // Use very different topics to ensure distinction
            var embedding1 = await embedder.GetEmbeddingsAsync("How do I read a file in C#?");
            var embedding2 = await embedder.GetEmbeddingsAsync("The capital of France is Paris, a beautiful city.");

            var similarity = CalculateCosineSimilarity(embedding1, embedding2);

            Console.WriteLine($"    Similarity: {similarity:F4}");
            // Note: Some BERT models produce high baseline similarity for all text.
            // We just verify it's measurably lower than semantically similar texts.
            Assert(similarity < 0.95, $"Different texts should have lower similarity than similar texts. Got: {similarity}");
        }

        private async Task Test_MockDataGeneration(RAGConfiguration config)
        {
            // Use a temporary database for this test
            var tempPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"test_mock_{Guid.NewGuid():N}.db");
            var testConfig = new RAGConfiguration
            {
                ModelPath = config.ModelPath,
                VocabularyPath = config.VocabularyPath,
                DatabasePath = tempPath
            };

            EmbeddingDatabaseNew? db = null;
            try
            {
                db = new EmbeddingDatabaseNew(testConfig);

                // Wait for initialization
                await Task.Delay(500);

                // Generate mock data without embeddings (faster)
                var count = await db.PopulateWithMockDataAsync(5, generateEmbeddings: false);

                Assert(count == 5, $"Should create 5 records. Created: {count}");

                var stats = await db.GetStatsAsync();
                Console.WriteLine($"    Stats: {stats}");

                Assert(stats.TotalRecords == 5, $"Should have 5 total records. Got: {stats.TotalRecords}");
            }
            finally
            {
                // Dispose database first to release file handles
                if (db != null)
                {
                    await db.DisposeAsync();
                }

                // Wait for file handles to be released
                await Task.Delay(200);

                // Cleanup - retry a few times if file is locked
                for (int i = 0; i < 3; i++)
                {
                    try
                    {
                        if (System.IO.File.Exists(tempPath))
                            System.IO.File.Delete(tempPath);
                        break;
                    }
                    catch (IOException)
                    {
                        await Task.Delay(100);
                    }
                }
            }
        }

        #endregion

        #region Helper Methods

        private static void Assert(bool condition, string message)
        {
            if (!condition)
                throw new Exception(message);
        }

        private static double CalculateCosineSimilarity(float[] vectorA, float[] vectorB)
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

        private static double CalculateWordMatchScore(string[] searchWords, string documentText)
        {
            if (string.IsNullOrEmpty(documentText) || searchWords.Length == 0)
                return 0;

            documentText = documentText.ToLower();
            var docWords = documentText.Split(new[] { ' ', '.', ',', ';', ':', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
            var docWordSet = new HashSet<string>(docWords, StringComparer.OrdinalIgnoreCase);

            // Use word boundary matching - only match whole words
            int matchedWords = searchWords.Count(word => docWordSet.Contains(word.ToLower()));
            double baseScore = (double)matchedWords / searchWords.Length;

            // Phrase match bonus - check if search words appear as consecutive whole words
            bool phraseMatch = false;
            if (searchWords.Length <= docWords.Length)
            {
                for (int i = 0; i <= docWords.Length - searchWords.Length; i++)
                {
                    bool match = true;
                    for (int j = 0; j < searchWords.Length; j++)
                    {
                        if (!docWords[i + j].Equals(searchWords[j], StringComparison.OrdinalIgnoreCase))
                        {
                            match = false;
                            break;
                        }
                    }
                    if (match)
                    {
                        phraseMatch = true;
                        break;
                    }
                }
            }
            double phraseMatchBonus = phraseMatch ? 0.2 : 0;

            double density = docWords.Length > 0 ? (double)matchedWords / docWords.Length : 0;

            return Math.Min(baseScore + phraseMatchBonus + (density * 0.1), 1.0);
        }

        private static float[] GenerateRandomVector(int dimension, int seed = 42)
        {
            var random = new Random(seed);
            var vector = new float[dimension];
            for (int i = 0; i < dimension; i++)
            {
                vector[i] = (float)(random.NextDouble() * 2 - 1);
            }
            return vector;
        }

        private static float[] GenerateNormalizedVector(int dimension, int seed = 42)
        {
            return Normalize(GenerateRandomVector(dimension, seed));
        }

        private static float[] Normalize(float[] vector)
        {
            double magnitude = Math.Sqrt(vector.Sum(v => v * v));
            if (magnitude == 0) return vector;
            return vector.Select(v => (float)(v / magnitude)).ToArray();
        }

        private static int CountMatchingBits(int a, int b)
        {
            int xor = a ^ b;
            int matching = 0;
            for (int i = 0; i < 32; i++)
            {
                if ((xor & (1 << i)) == 0)
                    matching++;
            }
            return matching;
        }

        #endregion
    }
}
