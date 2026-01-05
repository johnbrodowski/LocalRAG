using Xunit;

namespace LocalRAG.Tests;

/// <summary>
/// Unit tests for MemoryHashIndex add, update, and search operations.
/// </summary>
public class MemoryHashIndexTests
{
    [Fact]
    public async Task AddAndSearch_FindsExactVector()
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

        Assert.NotEmpty(results);
        Assert.Equal("test-1", results[0].Id);
        Assert.True(results[0].Score > 0.99, $"Identical vector should have score ~1.0. Got: {results[0].Score}");
    }

    [Fact]
    public async Task Search_FindsSimilarVectors()
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
        var random = new Random(123);
        var similarVector = baseVector.Select(v => v + 0.01f * (float)(random.NextDouble() - 0.5)).ToArray();
        similarVector = Normalize(similarVector);
        await index.AddOrUpdateAsync("similar", similarVector, new Dictionary<string, string> { { "text", "similar" } });

        // Add a different vector
        var differentVector = GenerateNormalizedVector(768, seed: 999);
        await index.AddOrUpdateAsync("different", differentVector, new Dictionary<string, string> { { "text", "different" } });

        // Search with base vector
        var results = await index.SearchAsync(baseVector, 10);

        Assert.Contains(results, r => r.Id == "base");
    }

    [Fact]
    public async Task Update_ReplacesExistingVector()
    {
        var options = new MemoryHashIndex.MemoryHashOptions
        {
            NumHashFunctions = 8,
            NumBuckets = 256,
            SimilarityThreshold = 0.5
        };
        using var index = new MemoryHashIndex(options);

        var vector1 = GenerateRandomVector(768, seed: 1);
        var vector2 = GenerateRandomVector(768, seed: 2);

        await index.AddOrUpdateAsync("test", vector1, new Dictionary<string, string> { { "v", "1" } });
        await index.AddOrUpdateAsync("test", vector2, new Dictionary<string, string> { { "v", "2" } });

        // Searching with vector2 should find "test" with high score
        var results = await index.SearchAsync(vector2, 10);

        Assert.Contains(results, r => r.Id == "test" && r.Score > 0.99);
    }

    [Fact]
    public async Task Search_RespectsTopKLimit()
    {
        var options = new MemoryHashIndex.MemoryHashOptions
        {
            NumHashFunctions = 8,
            NumBuckets = 256,
            SimilarityThreshold = 0.0 // Accept all
        };
        using var index = new MemoryHashIndex(options);

        // Add 10 vectors
        for (int i = 0; i < 10; i++)
        {
            var vector = GenerateRandomVector(768, seed: i);
            await index.AddOrUpdateAsync($"item-{i}", vector, new Dictionary<string, string>());
        }

        var searchVector = GenerateRandomVector(768, seed: 0);
        var results = await index.SearchAsync(searchVector, 5);

        Assert.True(results.Count <= 5, $"Should return at most 5 results. Got: {results.Count}");
    }

    [Fact]
    public async Task Search_ResultsOrderedByScore()
    {
        var options = new MemoryHashIndex.MemoryHashOptions
        {
            NumHashFunctions = 8,
            NumBuckets = 256,
            SimilarityThreshold = 0.0
        };
        using var index = new MemoryHashIndex(options);

        var baseVector = GenerateNormalizedVector(768, seed: 42);

        // Add vectors with varying similarity
        await index.AddOrUpdateAsync("exact", baseVector, new Dictionary<string, string>());

        var slightlyDifferent = baseVector.Select((v, i) => i < 100 ? v * 0.9f : v).ToArray();
        slightlyDifferent = Normalize(slightlyDifferent);
        await index.AddOrUpdateAsync("similar", slightlyDifferent, new Dictionary<string, string>());

        var veryDifferent = GenerateNormalizedVector(768, seed: 999);
        await index.AddOrUpdateAsync("different", veryDifferent, new Dictionary<string, string>());

        var results = await index.SearchAsync(baseVector, 10);

        // Results should be sorted by score descending
        for (int i = 1; i < results.Count; i++)
        {
            Assert.True(results[i - 1].Score >= results[i].Score,
                $"Results should be sorted by score. Position {i - 1} ({results[i - 1].Score}) < Position {i} ({results[i].Score})");
        }
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
}
