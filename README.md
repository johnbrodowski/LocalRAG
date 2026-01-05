# About this code
This example was extracted from AGPA — my fully autonomous general-purpose agent (closed-source, ~150k LOC).

# LocalRAG

A local Retrieval-Augmented Generation (RAG) system for .NET that uses BERT embeddings and multiple search strategies for efficient semantic search and information retrieval.

## Overview

LocalRAG provides a complete RAG implementation that runs entirely on your local machine, with no external API dependencies. It combines BERT-based embeddings with multiple search strategies to provide fast and accurate semantic search capabilities.

## Features

- **BERT-based Text Embeddings**: Uses ONNX Runtime for high-performance BERT inference
- **Multiple Search Strategies**:
  - Locality-Sensitive Hashing (LSH) for efficient similarity search
  - Full-Text Search (FTS5) integration via SQLite
  - Memory-based vector indexing for real-time queries
- **SQLite Database**: Persistent storage for embeddings and metadata
- **Configurable Processing**: Adjustable chunking, overlap, and threading parameters
- **Asynchronous API**: Non-blocking operations for better performance
- **Windows Forms Demo**: Example application demonstrating usage

## Prerequisites

- .NET 10.0 SDK or later
- Windows, Linux, or macOS
- BERT ONNX model (see setup instructions below)
- BERT vocabulary file (vocab.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LocalRAG.git
cd LocalRAG
```

2. Restore NuGet packages:
```bash
dotnet restore
```

3. Download a BERT model in ONNX format:
   - Visit [Hugging Face ONNX Models](https://huggingface.co/models?library=onnx&search=bert)
   - Models under Apache 2.0; see Hugging Face for details
   - Download a BERT model (e.g., `bert-base-uncased` or `bert-large-uncased`)
   - Place the `.onnx` file in the `onnxBERT/` directory
   - Download the corresponding `vocab.txt` file
   - Place it in the `Vocabularies/` directory

4. Build the project:
```bash
dotnet build
```

## Quick Start

```csharp
using LocalRAG;

// Configure the RAG system
var config = new RAGConfiguration
{
    ModelPath = "onnxBERT/model.onnx",
    VocabularyPath = "Vocabularies/vocab.txt",
    DatabasePath = "Database/embeddings.db"
};

// Initialize the database
using var database = new EmbeddingDatabaseNew(config);

// Add documents
await database.AddRequestToEmbeddingDatabaseAsync(
    requestId: "doc1",
    theRequest: "What is machine learning?",
    embed: true
);

await database.UpdateTextResponse(
    requestId: "doc1",
    message: "Machine learning is a subset of artificial intelligence...",
    embed: true
);

// Search for similar content
var results = await database.SearchEmbeddingsAsync(
    searchText: "artificial intelligence",
    topK: 5,
    minimumSimilarity: 0.75f
);

foreach (var result in results)
{
    Console.WriteLine($"Similarity: {result.Similarity:F3}");
    Console.WriteLine($"Request: {result.Request}");
    Console.WriteLine($"Response: {result.TextResponse}");
}
```

## Configuration

The `RAGConfiguration` class provides various settings:

```csharp
public class RAGConfiguration
{
    // File paths
    public string DatabasePath { get; set; }      // SQLite database location
    public string ModelPath { get; set; }         // ONNX model file
    public string VocabularyPath { get; set; }    // BERT vocab file

    // Embedding settings
    public int MaxSequenceLength { get; set; } = 512;
    public int WordsPerString { get; set; } = 40;
    public double OverlapPercentage { get; set; } = 15;

    // LSH settings
    public int NumberOfHashFunctions { get; set; } = 8;
    public int NumberOfHashTables { get; set; } = 10;

    // Performance settings
    public int InterOpNumThreads { get; set; } = 32;
    public int IntraOpNumThreads { get; set; } = 2;
    public int MaxCacheItems { get; set; } = 10000;
}
```

## Architecture

### Core Components

- **EmbedderClassNew**: Handles BERT embeddings generation using ONNX Runtime
- **EmbeddingDatabaseNew**: Main database interface with SQLite storage
- **MemoryHashIndex**: In-memory hash-based indexing for fast lookups
- **FeedbackDatabaseValues**: Data model for stored documents and embeddings

### Search Flow

1. Text is preprocessed (tokenized, stop words removed)
2. BERT generates embeddings via ONNX Runtime
3. Embeddings are indexed using LSH for fast retrieval
4. Multiple search strategies are combined for optimal results
5. Results are ranked by similarity score

## Demo Application

The `DemoApp` project provides a Windows Forms application demonstrating LocalRAG usage:

```bash
cd DemoApp
dotnet run
```

The demo shows:
- Adding documents with embeddings
- Searching for similar content
- Retrieving conversation history
- Formatting search results

## Performance Considerations

- **First Run**: Initial embedding generation may be slow
- **Caching**: Frequently accessed embeddings are cached in memory
- **Threading**: Adjust `InterOpNumThreads` and `IntraOpNumThreads` based on your CPU
- **Database Size**: SQLite performs well up to several million embeddings

## Running Tests

LocalRAG includes comprehensive test coverage with both unit and integration tests.

### Quick Start - Run All Tests

1. **One-time setup** - Configure your test environment:
   ```bash
   # Windows PowerShell
   copy test.runsettings.example test.runsettings

   # Linux/Mac
   cp test.runsettings.example test.runsettings
   ```

2. **Edit `test.runsettings`** and update these two paths:
   ```xml
   <BERT_MODEL_PATH>C:\path\to\your\model.onnx</BERT_MODEL_PATH>
   <BERT_VOCAB_PATH>C:\path\to\your\vocab.txt</BERT_VOCAB_PATH>
   ```

   **Important:**
   - Use **absolute paths** to the actual files (not directories)
   - Point to the `.onnx` file itself (e.g., `model2.onnx`)
   - Point to the `.txt` vocab file (e.g., `base_cased_large.txt`)

3. **Run tests:**
   ```bash
   dotnet test --settings test.runsettings
   ```

### Expected Output

Successful test run (all 38 tests passing):
```
Test summary: total: 38, failed: 0, succeeded: 38, skipped: 0, duration: 13.4s
Build succeeded in 15.3s
```

### Test Categories

#### Unit Tests (33 tests)
Fast tests that don't require BERT models. Run with:
```bash
dotnet test --filter "Category!=Integration"
```

These test core functionality:
- Text preprocessing and tokenization
- Database operations
- Search algorithms (LSH, FTS)
- Configuration handling

#### Integration Tests (5 tests)
Tests that require actual BERT models. Run with:
```bash
dotnet test --settings test.runsettings --filter "Category=Integration"
```

These test:
- BERT embedding generation
- End-to-end search with real embeddings
- Model dimension validation (768 for base, 1024 for large)
- Semantic similarity calculations
- Mock data generation with embeddings

### Visual Studio Users

In Visual Studio, the tests will appear in Test Explorer. To run integration tests:
1. Right-click on the solution in Solution Explorer
2. Select "Configure Run Settings" → "Select Solution Wide runsettings File"
3. Choose your `test.runsettings` file
4. Run tests normally from Test Explorer

### Troubleshooting Tests

#### All tests are skipped
```
Test summary: total: 38, failed: 0, succeeded: 0, skipped: 38
```
**Solution:** You need to create `test.runsettings` and run with `--settings test.runsettings`

#### Integration tests fail with "BERT model not configured"
```
LocalRAG.Tests.SkipException : BERT model not configured.
```
**Solutions:**
1. Verify `test.runsettings` exists in the project root
2. Check that paths point to files, not directories:
   - ❌ Wrong: `C:\...\onnxBERT\`
   - ✅ Correct: `C:\...\onnxBERT\model2.onnx`
3. Verify files exist at those paths:
   ```bash
   # Windows
   dir "C:\path\to\model.onnx"
   dir "C:\path\to\vocab.txt"

   # Linux/Mac
   ls -la /path/to/model.onnx
   ls -la /path/to/vocab.txt
   ```
4. Make sure you're running with: `dotnet test --settings test.runsettings`

#### Model file not found
Ensure you've downloaded a BERT model:
1. Visit [Hugging Face ONNX Models](https://huggingface.co/models?library=onnx&search=bert)
2. Download a BERT model (e.g., `bert-base-uncased` or `bert-large-uncased`)
3. Update `test.runsettings` with the actual file path

### CI/CD Integration

For continuous integration, set environment variables instead:
```bash
# Linux/Mac
export BERT_MODEL_PATH="/path/to/model.onnx"
export BERT_VOCAB_PATH="/path/to/vocab.txt"
dotnet test

# Windows
set BERT_MODEL_PATH=C:\path\to\model.onnx
set BERT_VOCAB_PATH=C:\path\to\vocab.txt
dotnet test
```

Or skip integration tests in CI:
```bash
dotnet test --filter "Category!=Integration"
```

## Troubleshooting

### Model not found error
Ensure the ONNX model file exists at the configured `ModelPath`. Download from Hugging Face if needed.

### Out of memory errors
Reduce `MaxCacheItems` or `MaxSequenceLength` in configuration.

### Slow embedding generation
- Use a smaller BERT model (base vs. large)
- Increase thread count if you have more CPU cores
- Enable GPU support via ONNX Runtime GPU packages

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache License 2.0 - see [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgments

- Built with [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- Uses [FastBertTokenizer](https://github.com/NMZivkovic/FastBertTokenizer) for tokenization
- BERT models from [Hugging Face](https://huggingface.co/)

## Roadmap

- [ ] GPU acceleration support
- [ ] More embedding models (Sentence Transformers, etc.)
- [ ] Vector database integration options
- [ ] REST API interface
- [ ] Multi-language support

## Support

For questions and issues, please open an issue on GitHub
