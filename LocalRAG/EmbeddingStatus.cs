using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnthropicApp.Database
{
    public record EmbeddingStatus
    {
        public bool RequestEmbedding { get; set; }
        public bool SummaryEmbedding { get; set; }
        public bool RequestEmbeddingList { get; set; }
        public bool TextResponseEmbedding { get; set; }
        public bool ToolUseTextResponseEmbedding { get; set; }
        public bool TextResponseEmbeddingList { get; set; }
        public bool ToolUseTextResponseEmbeddingList { get; set; }
        public bool SummaryEmbeddingList { get; set; }

    }

}
