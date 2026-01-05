using Azure.Core;

using LocalRAG;

using Microsoft.ML.OnnxRuntimeGenAI;

using System.Diagnostics;
using System.Text;

namespace DemoApp
{
    public partial class Form1 : Form
    {
        private EmbeddingDatabaseNew db;

        private string _currentRequest = string.Empty;

        public Form1()
        {
            InitializeComponent();
        }

        private async void Form1_Load(object sender, EventArgs e)
        {
            _feedbackEmbeddingDatabase = new EmbeddingDatabaseNew(new RAGConfiguration());
        }

        private async Task<string> FormatRelevantFeedback(string query, int topK = 30, int minWordsToTriggerLsh = 1)
        {
            var sb = new StringBuilder();

            //if (CountWords(query) > minWordsToTriggerLsh)
            //{
                List<FeedbackDatabaseValues> results = await db.SearchEmbeddingsAsync(
                searchText: query,
                topK: topK,
              //  minimumSimilarity: 0.05f,
                  minimumSimilarity: 0.00f,
                searchLevel: 2
                );

            void AppendSection(StringBuilder sb, string title, string? content)
            {
                if (!string.IsNullOrEmpty(content))
                {
                    sb.AppendLine($"{title}: {content}");
                }
            }

            var sortedResults = results
                .Where(x => x.Similarity.HasValue)
                .OrderByDescending(x => x.Similarity)
            .ToList();

            sb.AppendLine($"# Here is a list of potentially relevant feedback from the RAG for you to consider when formulating your response.");
            sb.AppendLine(@"Note: Similarity scores are guides, not gospel. Evaluate actual relevance and usefulness of returned content, regardless of numerical scores.");

            foreach (var (result, index) in sortedResults.Select((result, index) => (result, index + 1)))
            {
                AppendSection(sb, "Users Request", result.Request);

                sb.AppendLine().AppendLine($"Results: {index} - Similarity {result.Similarity:F5}");

                AppendSection(sb, "Users Request", result.Request);
                AppendSection(sb, "AI text Response", GetTruncatedText(result.TextResponse, 3000));
                AppendSection(sb, "AI Tool Use TextResponse", GetTruncatedText(result.ToolUseTextResponse, 3000));
            }

            sb.AppendLine($"User Request:");
            sb.AppendLine();

            return sb.ToString();
        }

        private async Task LoadConversationHistory(int maxMsgCount = 10, bool showToolInputs = true, bool removeToolMessages = true)
        {
            try
            {
                List<(string request, string textResponse, string toolUseTextResponse, string toolContent, string toolResult, string requestID)> recentHistory =

                await db.GetConversationHistoryAsync(maxMsgCount);

                foreach (var (request, textResponse, toolUseTextResponse, toolContent, toolResult, requestId) in recentHistory)
                {
                    if (!string.IsNullOrEmpty(request) && !string.IsNullOrEmpty(textResponse))
                    {
                        Debug.WriteLine($"Request: {request}");
                        Debug.WriteLine($"Response: {textResponse}");
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error loading history: {ex.Message}");
            }
        }

        public int CountWords(string str)
        {
            if (string.IsNullOrWhiteSpace(str))
                return 0;

            string[] words = str.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            return words.Length;
        }

        private string GetTruncatedText(string text, int maxLength, string truncationMessage = "")
        {
            if (string.IsNullOrEmpty(text) || text.Length <= maxLength)
                return text;

            return text.Substring(0, maxLength) + "... " + truncationMessage;
        }

        private async void btnSearch_Click(object sender, EventArgs e)
        {
            var msg = await FormatRelevantFeedback(txtQuery.Text);
            txtResult.Text = msg;
        }

        private async void btnRunTests_Click(object sender, EventArgs e)
        {
            txtResult.Text = "Running tests...\r\n";
            btnRunTests.Enabled = false;

            try
            {
                var tests = new EmbeddingTests();
                var sb = new StringBuilder();

                // Capture console output
                var originalOut = Console.Out;
                using var writer = new StringWriter();
                Console.SetOut(writer);

                try
                {
                    // Run unit tests (no BERT model required)
                    var results = await tests.RunAllTestsAsync(null);

                    // Try to run integration tests if config is available
                    try
                    {
                        var config = new RAGConfiguration();
                        if (System.IO.File.Exists(config.ModelPath))
                        {
                            sb.AppendLine("\r\n--- Running Integration Tests ---\r\n");
                            results = await tests.RunAllTestsAsync(config);
                        }
                    }
                    catch (Exception ex)
                    {
                        sb.AppendLine($"\r\nIntegration tests skipped: {ex.Message}");
                    }
                }
                finally
                {
                    Console.SetOut(originalOut);
                }

                sb.Insert(0, writer.ToString().Replace("\n", "\r\n"));
                txtResult.Text = sb.ToString();
            }
            catch (Exception ex)
            {
                txtResult.Text = $"Test error: {ex.Message}\r\n{ex.StackTrace}";
            }
            finally
            {
                btnRunTests.Enabled = true;
            }
        }

        private async void btnGenerateMockData_Click(object sender, EventArgs e)
        {
            btnGenerateMockData.Enabled = false;
            txtResult.Text = "Generating mock data...\r\n";

            try
            {
                var sb = new StringBuilder();
                sb.AppendLine("=== Mock Data Generation ===\r\n");

                // Get stats before
                var statsBefore = await _feedbackEmbeddingDatabase.GetStatsAsync();
                sb.AppendLine($"Before: {statsBefore.TotalRecords} records\r\n");

                // Generate mock data
                int count = await _feedbackEmbeddingDatabase.PopulateWithMockDataAsync(
                    count: 20,
                    generateEmbeddings: true,
                    progress: (done, total) =>
                    {
                        this.Invoke(() => txtResult.Text = $"Generating: {done}/{total}...");
                    }
                );

                // Get stats after
                var statsAfter = await _feedbackEmbeddingDatabase.GetStatsAsync();
                sb.AppendLine($"Created {count} mock records\r\n");
                sb.AppendLine($"After: {statsAfter}\r\n");

                txtResult.Text = sb.ToString();
            }
            catch (Exception ex)
            {
                txtResult.Text = $"Error: {ex.Message}\r\n{ex.StackTrace}";
            }
            finally
            {
                btnGenerateMockData.Enabled = true;
            }
        }
    }
}
