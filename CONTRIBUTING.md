# Contributing to LocalRAG

Thank you for your interest in contributing to LocalRAG! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, .NET version, etc.)
- Any relevant error messages or stack traces

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the proposed feature
- Why this feature would be useful
- Potential implementation approaches (if applicable)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the coding guidelines below
3. **Test your changes** thoroughly
4. **Update documentation** if needed (README, code comments, etc.)
5. **Commit your changes** with clear, descriptive commit messages
6. **Push to your fork** and submit a pull request

#### Pull Request Guidelines

- Keep changes focused - one feature/fix per PR
- Include tests for new functionality
- Ensure all tests pass
- Update documentation as needed
- Follow existing code style and conventions

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/LocalRAG.git
cd LocalRAG
```

2. Restore dependencies:
```bash
dotnet restore
```

3. Build the project:
```bash
dotnet build
```

4. Run tests (if available):
```bash
dotnet test
```

## Coding Guidelines

### General Principles

- Write clear, readable code
- Add XML documentation comments to public APIs
- Use meaningful variable and method names
- Keep methods focused and reasonably sized
- Handle errors appropriately

### C# Style

- Use C# naming conventions (PascalCase for public members, camelCase for private fields with _ prefix)
- Enable nullable reference types and handle nulls appropriately
- Prefer `async`/`await` for I/O operations
- Use `using` statements for IDisposable resources
- Keep line length reasonable (under 120 characters when possible)

### Example

```csharp
/// <summary>
/// Retrieves embeddings for the specified text.
/// </summary>
/// <param name="text">The text to embed.</param>
/// <returns>A task representing the embedding operation.</returns>
public async Task<float[]> GetEmbeddingsAsync(string text)
{
    if (string.IsNullOrWhiteSpace(text))
        throw new ArgumentException("Text cannot be null or empty.", nameof(text));

    return await GetEmbeddingsInternalAsync(text);
}
```

### Documentation

- Add XML comments to all public types and members
- Keep README.md up to date with API changes
- Document any breaking changes in commit messages

## Testing

- Write unit tests for new functionality
- Ensure existing tests pass before submitting
- Test edge cases and error conditions
- Include integration tests for complex features

## Commit Messages

Write clear, descriptive commit messages:

- Use the imperative mood ("Add feature" not "Added feature")
- Keep the first line under 50 characters
- Provide detailed explanation in the body if needed
- Reference issues/PRs when applicable

Example:
```
Add support for custom BERT models

- Allow users to specify different BERT architectures
- Update configuration to handle variable hidden dimensions
- Add validation for model compatibility

Fixes #123
```

## License

By contributing to LocalRAG, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

If you have questions about contributing, feel free to open an issue with the "question" label.

Thank you for contributing to LocalRAG!
