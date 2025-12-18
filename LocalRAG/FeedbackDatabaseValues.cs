using Newtonsoft.Json;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalRAG
{
    public class FeedbackDatabaseValues
    {
        public int Id { get; set; }
        public string? RequestID { get; set; }
        public string? UserMessageType { get; set; } = "text";
        public string? AssistantMessageType { get; set; } = "text";
        public int? Rating { get; set; }
        public string? MetaData { get; set; }
        public string? Request { get; set; }
        public string? TextResponse { get; set; }
        public string? ToolUseTextResponse { get; set; }
        public string? ToolName { get; set; }
        public string? ToolContent { get; set; }
        public string? ToolResult { get; set; }
        public string? Code { get; set; }
        public string? Summary { get; set; }
        public string? Comment { get; set; }
        public string? Errors { get; set; }
        public string? RequestStatus { get; set; }
        public DateTime? Timestamp { get; set; }
        public double? Similarity { get; set; }

        // Embeddings
        public float[]? RequestEmbedding { get; set; }
        public List<float[]>? RequestEmbeddingList { get; set; }
        public float[]? TextResponseEmbedding { get; set; }
        public List<float[]>? TextResponseEmbeddingList { get; set; }
        public float[]? ToolUseTextResponseEmbedding { get; set; }
        public List<float[]>? ToolUseTextResponseEmbeddingList { get; set; }
        public float[]? SummaryEmbedding { get; set; }
        public List<float[]>? SummaryEmbeddingList { get; set; }

        // Embedding flags
        public bool Embed { get; set; }
        public bool EmbedRequest { get; set; }
        public bool EmbedRequestList { get; set; }
        public bool EmbedTextResponse { get; set; }
        public bool EmbedTextResponseList { get; set; }
        public bool EmbedToolUseTextResponse { get; set; }
        public bool EmbedToolUseTextResponseList { get; set; }
        public bool EmbedSummary { get; set; }
        public bool EmbedSummaryList { get; set; }

        // Formatted fields for appending
        public string? newComment { get; set; }
        public string? newToolContent { get; set; }
        public string? newToolResult { get; set; }
        public string? newTextResponse { get; set; }
        public string? newToolUseTextResponse { get; set; }
        public string? newToolName { get; set; }

         
        [JsonProperty("Feedback", NullValueHandling = NullValueHandling.Ignore)]
        public string? Feedback { get; set; }
         
        [JsonProperty("OK", NullValueHandling = NullValueHandling.Ignore)]
        public string? OK { get; set; }

        [JsonProperty("ERROR", NullValueHandling = NullValueHandling.Ignore)]
        public string? ERROR { get; set; }

        [JsonProperty("REPORT", NullValueHandling = NullValueHandling.Ignore)]
        public string? REPORT { get; set; }

        [JsonProperty("MSG", NullValueHandling = NullValueHandling.Ignore)]
        public string? MSG { get; set; }

        [JsonProperty("ConnectionString", NullValueHandling = NullValueHandling.Ignore)]
        public string? ConnectionString { get; set; }
    }

}
