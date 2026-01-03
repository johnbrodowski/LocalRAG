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
            var config = new RAGConfiguration
            {
                DatabasePath = "Database/Memory/FreshEmbeddings.db"  // New file
            };

            var db = new EmbeddingDatabaseNew(config);

            // Option 1: Generate mock data WITH embeddings (will use CPU/RAM)
            //await db.PopulateWithMockDataAsync(
            //    count: 50,
            //    generateEmbeddings: true,
            //    progress: (done, total) => Debug.WriteLine($"[{done}/{total}]")
            //);

            // Option 2: Generate mock data WITHOUT embeddings first (fast)
            //await db.PopulateWithMockDataAsync(count: 50, generateEmbeddings: false);
            // Then backfill separately
            await db.BackfillEmbeddingsAsync("all");

            // Check stats
            var stats = await db.GetStatsAsync();
            Debug.WriteLine(stats);
            // Output: Records: 50 total, 50 with embeddings, 0 missing
            //         LSH: 10 buckets, 150 entries

            // If you need to start over
           // await db.ClearAllDataAsync();









            // Option 1: Only fill missing embeddings (fastest, keeps existing)
            //var result1 = await db.BackfillEmbeddingsAsync("missing");
            //Debug.WriteLine($"Missing:{result1}\n\n");




            // Option 2: Replace ALL embeddings (use this since old ones are bad)
            //var result2 = await db.BackfillEmbeddingsAsync("all");
            //Debug.WriteLine($"All:{result2}\n\n");

            // Option 3: With progress tracking
            //var result3 = await db.BackfillEmbeddingsAsync(
            //    mode: "missing",
            //    batchSize: 2,
            //    progress: (processed, total, requestId) =>
            //    {
            //        Debug.WriteLine($"[{processed}/{total}] Processing: {requestId}");
            //    }
            //);

            // Option 4: With cancellation support
            //var cts = new CancellationTokenSource();
            //var result4 = await db.BackfillEmbeddingsAsync("\n\nAll w/CancellationToken", cancellationToken: cts.Token);

            //// Check results
            //Debug.WriteLine(result4);
            // Output: "Backfill: 150/200 succeeded, 2 failed, 48 skipped in 45.3s"
 

            // await Test();
        }

        private async Task Test()
        {
            //await _feedbackEmbeddingDatabase.AddRequestToEmbeddingDatabaseAsync(
            // requestId: "1000",
            // theRequest: "a sample message",
            // embed: true
            // );

            //await _feedbackEmbeddingDatabase.UpdateTextResponse(
            //    requestId: "1000",
            //    message: "a sample response",
            //    embed: true
            //    );


            //   await LoadConversationHistory();

            //var msg = await FormatRelevantFeedback();

            //Debug.WriteLine($"\n\n\n\n{msg}");


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

                // string truncatedMessage = $"Long messages are truncated for efficiency, use the
                // 'request_full_content' tool with the Request ID to full content.";

              //  if (!string.IsNullOrEmpty(query) && results.Count > 0)
               // {
                    sb.AppendLine($"# Here is a list of potentially relevant feedback from the RAG for you to consider when formulating your response.");
                    sb.AppendLine(@"Note: Similarity scores are guides, not gospel. Evaluate actual relevance and usefulness of returned content, regardless of numerical scores.");
                    // sb.AppendLine(@"Attention: Long responses will be truncated, but soon you
                    // will be able to request the complete response if you need to.");
                    foreach (var (result, index) in sortedResults.Select((result, index) => (result, index + 1)))
                    {
                
                    AppendSection(sb, "Users Request", result.Request);
                
                    sb.AppendLine().AppendLine($"Results: {index} - Similarity {result.Similarity:F5}");
                        //.AppendLine($"# Similarity #\n{result.Similarity:F5}");

                        // AppendSection(sb, "Request ID", result.RequestID);
                        //AppendSection(sb, "Rating", result.Rating?.ToString() ?? "0");
                       // if (!string.IsNullOrEmpty(result.Request)) 
                    AppendSection(sb, "Users Request", result.Request);

                       // if (!string.IsNullOrEmpty(result.TextResponse)) 
                    AppendSection(sb, "AI text Response", GetTruncatedText(result.TextResponse, 3000));

                       // if (!string.IsNullOrEmpty(result.ToolUseTextResponse)) 
                    AppendSection(sb, "AI Tool Use TextResponse", GetTruncatedText(result.ToolUseTextResponse, 3000));

                        //AppendSection(sb, "Tool Content", GetTruncatedText(result.ToolContent, 300));
                    }

                    sb.AppendLine($"User Request:");
                    sb.AppendLine();
                //}

                //sb.AppendLine($"{query}");
            //}
            //else
            //{
            //    sb.Append($"{query}");
           // }




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
                        //dbMessageList.Add(MessageAnthropic.CreateUserMessage(GetTruncatedText(request, 400, " - (Truncated)")));
                        //dbMessageList.Add(MessageAnthropic.CreateAssistantTextMessage(GetTruncatedText(textResponse ?? "N/A", 400, " - (Truncated)")));
                        // dbMessageList.Add(MessageAnthropic.CreateUserMessage(request));
                        //  dbMessageList.Add(MessageAnthropic.CreateAssistantTextMessage(textResponse));

                        Debug.WriteLine($"Request: {request}");
                        Debug.WriteLine($"Response: {textResponse}");

                    }
                }


            }
            catch (Exception ex)
            {

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





    }
}