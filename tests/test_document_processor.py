"""Test the document processor."""


def create_test_document():
    """Create a simple test document for testing."""
    import tempfile

    # Create a test text file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("""# Research Paper on Climate Change

## Abstract
This paper discusses the impact of climate change on global ecosystems. We analyze data from multiple sources and present findings about temperature trends.

## Authors
John Smith, Jane Doe

## Introduction  
Climate change is one of the most pressing issues of our time. The evidence shows that global temperatures are rising at an unprecedented rate.

DOI: 10.1234/climate.2023.001

## Methodology
We collected data from 50 weather stations across 10 countries over a 20-year period.

## Results
Our analysis shows a clear warming trend of 0.2°C per decade.

## Conclusion
Immediate action is needed to address climate change and its impacts on ecosystems.

## References
1. Smith et al. (2020). Climate patterns. Nature Climate Change.
2. Johnson et al. (2021). Global warming effects. Science.
""")
        return f.name


def test_document_processor():
    """Test the document processor with a text file."""
    from research_agent.processors.document_processor import DocumentProcessor

    # Create test document
    test_file = create_test_document()

    try:
        # Initialize processor
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)

        # Process document
        doc = processor.process_document(test_file)

        print("✅ Document Processing Test Results:")
        print(f"Title: {doc.title}")
        print(f"Authors: {doc.authors}")
        print(f"Content length: {len(doc.content)} chars")
        print(f"Number of chunks: {len(doc.chunks)}")
        print(f"Metadata keys: {list(doc.metadata.keys())}")

        # Show first chunk
        if doc.chunks:
            print("\nFirst chunk preview:")
            print(f"Text: {doc.chunks[0].text[:150]}...")
            print(f"Chunk metadata: {list(doc.chunks[0].metadata.keys())}")

        # Assertions
        assert doc.title
        assert doc.content
        assert doc.chunks
        assert doc.metadata.get("doi") == "10.1234/climate.2023.001"

    finally:
        # Clean up
        import os

        os.unlink(test_file)


if __name__ == "__main__":
    test_document_processor()
