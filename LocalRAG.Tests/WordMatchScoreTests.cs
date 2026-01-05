using Xunit;

namespace LocalRAG.Tests;

/// <summary>
/// Unit tests for word matching and scoring logic.
/// Tests proper word boundary detection to avoid substring false positives.
/// </summary>
public class WordMatchScoreTests
{
    [Fact]
    public void ExactWordMatch_ReturnsPositiveScore()
    {
        var searchWords = new[] { "the", "cat" };
        var document = "The cat sat on the mat";

        var score = CalculateWordMatchScore(searchWords, document);

        Assert.True(score > 0, $"Should find 'the' and 'cat'. Score: {score}");
    }

    [Fact]
    public void NoSubstringMatch_ReturnsZero()
    {
        var searchWords = new[] { "the" };
        var document = "brother together other";

        var score = CalculateWordMatchScore(searchWords, document);

        // With proper word boundary matching, "the" should NOT match in these words
        Assert.Equal(0, score);
    }

    [Fact]
    public void CaseInsensitive_MatchesRegardlessOfCase()
    {
        var searchWords = new[] { "hello", "world" };
        var document = "HELLO World";

        var score = CalculateWordMatchScore(searchWords, document);

        Assert.True(score > 0, "Should match case-insensitively");
    }

    [Fact]
    public void PhraseBonus_WhenWordsAreConsecutive()
    {
        var searchWords = new[] { "quick", "brown" };
        var documentWithPhrase = "The quick brown fox";
        var documentScattered = "Brown leaves fall in quick succession";

        var scorePhrase = CalculateWordMatchScore(searchWords, documentWithPhrase);
        var scoreScattered = CalculateWordMatchScore(searchWords, documentScattered);

        // Both should find the words, but phrase should score higher due to bonus
        Assert.True(scorePhrase > scoreScattered,
            $"Phrase match ({scorePhrase}) should score higher than scattered ({scoreScattered})");
    }

    [Fact]
    public void EmptyDocument_ReturnsZero()
    {
        var searchWords = new[] { "test" };

        Assert.Equal(0, CalculateWordMatchScore(searchWords, ""));
        Assert.Equal(0, CalculateWordMatchScore(searchWords, null!));
    }

    [Fact]
    public void EmptySearchWords_ReturnsZero()
    {
        var document = "Some text here";

        Assert.Equal(0, CalculateWordMatchScore(Array.Empty<string>(), document));
    }

    [Fact]
    public void PartialWordMatch_NotCounted()
    {
        var searchWords = new[] { "cat" };
        var document = "concatenate category scatter";

        var score = CalculateWordMatchScore(searchWords, document);

        Assert.Equal(0, score);
    }

    [Fact]
    public void PunctuationSeparated_StillMatches()
    {
        var searchWords = new[] { "hello", "world" };
        var document = "Hello, world! How are you?";

        var score = CalculateWordMatchScore(searchWords, document);

        Assert.True(score > 0, "Should match words separated by punctuation");
    }

    [Theory]
    [InlineData("test", "This is a test", true)]
    [InlineData("test", "Testing one two three", false)]
    [InlineData("is", "This is a test", true)]
    [InlineData("is", "This island is nice", true)] // "is" appears as whole word too
    [InlineData("and", "android sandbox", false)]
    public void WordBoundary_TheoryTests(string searchWord, string document, bool shouldMatch)
    {
        var searchWords = new[] { searchWord };
        var score = CalculateWordMatchScore(searchWords, document);

        if (shouldMatch)
            Assert.True(score > 0, $"'{searchWord}' should match in '{document}'");
        else
            Assert.Equal(0, score);
    }

    [Fact]
    public void MultipleOccurrences_CountedOnce()
    {
        var searchWords = new[] { "the" };
        var document = "The the the the the";

        var score = CalculateWordMatchScore(searchWords, document);

        // Score should be 1.0 (baseScore) + potential bonuses
        // but not multiplied by occurrence count
        Assert.True(score <= 1.0, $"Score should not exceed 1.0. Got: {score}");
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
}
