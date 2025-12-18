using AnthropicApp.Database;

using System.Diagnostics;
using System.Text;

namespace DemoApp
{
    public partial class Form1 : Form
    {
        private EmbeddingDatabaseNew _feedbackEmbeddingDatabase;

        private string _currentRequest = string.Empty;

        public Form1()
        {
            InitializeComponent();
        }

        private async void Form1_Load(object sender, EventArgs e)
        {
            _feedbackEmbeddingDatabase = new EmbeddingDatabaseNew(new RAGConfiguration());
             await Test();
        }

        private async Task Test()
        {
            await _feedbackEmbeddingDatabase.AddRequestToEmbeddingDatabaseAsync(
             requestId: "1000",
             theRequest: "a sample message",
             embed: true
             );

            await _feedbackEmbeddingDatabase.UpdateTextResponse(
                requestId: "1000",
                message: "a sample response",
                embed: true
                );


            await LoadConversationHistory();

            var msg = FormatRelevantFeedback();

            Debug.WriteLine($"{msg}");


        }

        private async Task<string> FormatRelevantFeedback(string userRequest = "sample message", int topK = 3, int minWordsToTriggerLsh = 1)
        {
            var sb = new StringBuilder();

            if (CountWords(userRequest) > minWordsToTriggerLsh)
            {
                List<FeedbackDatabaseValues> results = await _feedbackEmbeddingDatabase.SearchEmbeddingsAsync(
                    searchText: userRequest,
                    topK: topK,
                    // minimumSimilarity: 0.75f,
                    minimumSimilarity: 0.05f,
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

                if (!string.IsNullOrEmpty(userRequest) && results.Count > 0)
                {
                    sb.AppendLine($"# Here is a list of potentially relevant feedback from the RAG for you to consider when formulating your response.");
                    sb.AppendLine(@"Note: Similarity scores are guides, not gospel. Evaluate actual relevance and usefulness of returned content, regardless of numerical scores.");
                    // sb.AppendLine(@"Attention: Long responses will be truncated, but soon you
                    // will be able to request the complete response if you need to.");
                    foreach (var (result, index) in sortedResults.Select((result, index) => (result, index + 1)))
                    {
                        sb.AppendLine($"\nResults: {index} - Similarity {result.Similarity:F5}");
                        //.AppendLine($"# Similarity #\n{result.Similarity:F5}");

                        // AppendSection(sb, "Request ID", result.RequestID);
                        //AppendSection(sb, "Rating", result.Rating?.ToString() ?? "0");
                        if (!string.IsNullOrEmpty(result.Request)) AppendSection(sb, "Users Request", result.Request);

                        if (!string.IsNullOrEmpty(result.TextResponse)) AppendSection(sb, "AI text Response", GetTruncatedText(result.TextResponse, 3000));

                        if (!string.IsNullOrEmpty(result.ToolUseTextResponse)) AppendSection(sb, "AI Tool Use TextResponse", GetTruncatedText(result.ToolUseTextResponse, 3000));

                        //AppendSection(sb, "Tool Content", GetTruncatedText(result.ToolContent, 300));
                    }

                    sb.AppendLine($"User Request:");
                    sb.AppendLine();
                }

                sb.AppendLine($"{userRequest}");
            }
            else
            {
                sb.Append($"{userRequest}");
            }
            return sb.ToString();
        }

        private async Task LoadConversationHistory(int maxMsgCount = 10, bool showToolInputs = true, bool removeToolMessages = true)
        {
 

            try
            {
                List<(string request, string textResponse, string toolUseTextResponse, string toolContent, string toolResult, string requestID)> recentHistory =

                await _feedbackEmbeddingDatabase.GetConversationHistoryAsync(maxMsgCount);

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
    }
}