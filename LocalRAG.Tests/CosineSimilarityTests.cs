using Xunit;

namespace LocalRAG.Tests;

/// <summary>
/// Unit tests for cosine similarity calculations.
/// </summary>
public class CosineSimilarityTests
{
    [Fact]
    public void IdenticalVectors_ReturnOne()
    {
        var vectorA = new float[] { 1, 2, 3, 4, 5 };
        var vectorB = new float[] { 1, 2, 3, 4, 5 };

        var similarity = CalculateCosineSimilarity(vectorA, vectorB);

        Assert.True(Math.Abs(similarity - 1.0) < 0.0001, $"Expected 1.0, got {similarity}");
    }

    [Fact]
    public void OrthogonalVectors_ReturnZero()
    {
        var vectorA = new float[] { 1, 0, 0 };
        var vectorB = new float[] { 0, 1, 0 };

        var similarity = CalculateCosineSimilarity(vectorA, vectorB);

        Assert.True(Math.Abs(similarity) < 0.0001, $"Expected 0.0, got {similarity}");
    }

    [Fact]
    public void OppositeVectors_ReturnNegativeOne()
    {
        var vectorA = new float[] { 1, 2, 3 };
        var vectorB = new float[] { -1, -2, -3 };

        var similarity = CalculateCosineSimilarity(vectorA, vectorB);

        Assert.True(Math.Abs(similarity - (-1.0)) < 0.0001, $"Expected -1.0, got {similarity}");
    }

    [Fact]
    public void SimilarVectors_ReturnHighSimilarity()
    {
        var vectorA = new float[] { 1, 2, 3, 4, 5 };
        var vectorB = new float[] { 1.1f, 2.1f, 3.1f, 4.1f, 5.1f };

        var similarity = CalculateCosineSimilarity(vectorA, vectorB);

        Assert.True(similarity > 0.99, $"Expected > 0.99, got {similarity}");
    }

    [Fact]
    public void NullVectors_ReturnZero()
    {
        var similarity1 = CalculateCosineSimilarity(null!, new float[] { 1, 2, 3 });
        var similarity2 = CalculateCosineSimilarity(new float[] { 1, 2, 3 }, null!);

        Assert.Equal(0, similarity1);
        Assert.Equal(0, similarity2);
    }

    [Fact]
    public void EmptyVectors_ReturnZero()
    {
        var similarity = CalculateCosineSimilarity(Array.Empty<float>(), Array.Empty<float>());

        Assert.Equal(0, similarity);
    }

    [Fact]
    public void DifferentLengthVectors_UsesMinLength()
    {
        var vectorA = new float[] { 1, 0, 0 };
        var vectorB = new float[] { 1, 0 }; // Shorter

        var similarity = CalculateCosineSimilarity(vectorA, vectorB);

        // Should compute similarity using first 2 elements only
        Assert.True(similarity > 0.99, $"Expected high similarity for matching prefix, got {similarity}");
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
