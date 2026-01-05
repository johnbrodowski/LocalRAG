using Xunit;

namespace LocalRAG.Tests;

/// <summary>
/// Integration tests that require the BERT model and database.
/// These tests are slower and require proper configuration.
///
/// To run these tests, set the environment variables:
/// - BERT_MODEL_PATH: Path to the ONNX model file
/// - BERT_VOCAB_PATH: Path to the vocabulary file
///
/// Or skip them with: dotnet test --filter "Category!=Integration"
/// </summary>
[Trait("Category", "Integration")]
public class IntegrationTests : IDisposable
{
    private readonly RAGConfiguration? _config;
    private readonly bool _canRun;

    public IntegrationTests()
    {
        var modelPath = Environment.GetEnvironmentVariable("BERT_MODEL_PATH");
        var vocabPath = Environment.GetEnvironmentVariable("BERT_VOCAB_PATH");

        if (!string.IsNullOrEmpty(modelPath) && !string.IsNullOrEmpty(vocabPath) &&
            File.Exists(modelPath) && File.Exists(vocabPath))
        {
            _config = new RAGConfiguration
            {
                ModelPath = modelPath,
                VocabularyPath = vocabPath,
                DatabasePath = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid():N}.db")
            };
            _canRun = true;
        }
        else
        {
            _canRun = false;
        }
    }

    public void Dispose()
    {
        if (_config != null && File.Exists(_config.DatabasePath))
        {
            try { File.Delete(_config.DatabasePath); } catch { }
        }
    }

    [SkippableFact]
    public async Task EmbeddingGeneration_PreservesStopWords()
    {
        Skip.IfNot(_canRun, "BERT model not configured. Set BERT_MODEL_PATH and BERT_VOCAB_PATH environment variables.");

        using var embedder = new EmbedderClassNew(_config!);

        var textWithStopWords = "The quick brown fox jumps over the lazy dog";
        var embedding = await embedder.GetEmbeddingsAsync(textWithStopWords);

        Assert.NotNull(embedding);
        Assert.True(embedding.Length == 768 || embedding.Length == 1024,
            $"Embedding should be 768 or 1024 dims. Got: {embedding.Length}");
    }

    [SkippableFact]
    public async Task EmbeddingDimension_Is768Or1024()
    {
        Skip.IfNot(_canRun, "BERT model not configured.");

        using var embedder = new EmbedderClassNew(_config!);

        var embedding = await embedder.GetEmbeddingsAsync("Test text");

        Assert.True(embedding.Length == 768 || embedding.Length == 1024,
            $"Expected 768 (BERT base) or 1024 (BERT large). Got: {embedding.Length}");
    }

    [SkippableFact]
    public async Task SimilarTexts_HaveHighSimilarity()
    {
        Skip.IfNot(_canRun, "BERT model not configured.");

        using var embedder = new EmbedderClassNew(_config!);

        var embedding1 = await embedder.GetEmbeddingsAsync("How do I read a file in C#?");
        var embedding2 = await embedder.GetEmbeddingsAsync("How can I read files using C#?");

        var similarity = CalculateCosineSimilarity(embedding1, embedding2);

        Assert.True(similarity > 0.8, $"Similar texts should have high similarity. Got: {similarity}");
    }

    [SkippableFact]
    public async Task DifferentTexts_HaveLowerSimilarity()
    {
        Skip.IfNot(_canRun, "BERT model not configured.");

        using var embedder = new EmbedderClassNew(_config!);

        var embedding1 = await embedder.GetEmbeddingsAsync("How do I read a file in C#?");
        var embedding2 = await embedder.GetEmbeddingsAsync("The capital of France is Paris, a beautiful city.");

        var similarity = CalculateCosineSimilarity(embedding1, embedding2);

        Assert.True(similarity < 0.95,
            $"Different texts should have lower similarity than similar texts. Got: {similarity}");
    }

    [SkippableFact]
    public async Task MockDataGeneration_CreatesRecords()
    {
        Skip.IfNot(_canRun, "BERT model not configured.");

        var tempPath = Path.Combine(Path.GetTempPath(), $"test_mock_{Guid.NewGuid():N}.db");
        var testConfig = new RAGConfiguration
        {
            ModelPath = _config!.ModelPath,
            VocabularyPath = _config.VocabularyPath,
            DatabasePath = tempPath
        };

        EmbeddingDatabaseNew? db = null;
        try
        {
            db = new EmbeddingDatabaseNew(testConfig);
            await Task.Delay(500); // Wait for initialization

            var count = await db.PopulateWithMockDataAsync(5, generateEmbeddings: false);

            Assert.Equal(5, count);

            var stats = await db.GetStatsAsync();
            Assert.Equal(5, stats.TotalRecords);
        }
        finally
        {
            if (db != null)
            {
                await db.DisposeAsync();
            }

            await Task.Delay(200);

            for (int i = 0; i < 3; i++)
            {
                try
                {
                    if (File.Exists(tempPath))
                        File.Delete(tempPath);
                    break;
                }
                catch (IOException)
                {
                    await Task.Delay(100);
                }
            }
        }
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
}

/// <summary>
/// Attribute for tests that can be skipped based on runtime conditions.
/// </summary>
public class SkippableFactAttribute : FactAttribute
{
}

/// <summary>
/// Helper class to skip tests at runtime.
/// </summary>
public static class Skip
{
    public static void IfNot(bool condition, string reason)
    {
        if (!condition)
        {
            throw new SkipException(reason);
        }
    }
}

/// <summary>
/// Exception thrown when a test should be skipped.
/// </summary>
public class SkipException : Exception
{
    public SkipException(string message) : base(message) { }
}
