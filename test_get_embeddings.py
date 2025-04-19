# tests/test_embeddings.py
import pytest
from unittest.mock import Mock, patch
from Unknown import get_embeddings, get_openai_client  # Replace 'Unknown' with the actual module name

# Mock the get_openai_client function to return a mock client object
@pytest.fixture
def mock_client():
    client = Mock()
    client.embeddings.create.return_value = Mock(data=[{'embedding': [1.0, 2.0, 3.0]}])
    return client

# Mock the connect_openai function to return the mock client object
@pytest.fixture
def mock_connect_openai():
    with patch('Unknown.connect_openai') as mock_connect_openai:
        mock_connect_openai.return_value = Mock()
        yield mock_connect_openai

# Test case 1: Normal functionality with typical input
def test_get_embeddings_normal(mock_client, mock_connect_openai):
    """
    Test the get_embeddings function with a typical input.
    """
    text = "This is a typical text"
    result = get_embeddings(text)
    assert result == [1.0, 2.0, 3.0]
    mock_connect_openai.assert_called_once()
    client = mock_connect_openai.return_value
    client.embeddings.create.assert_called_once_with(
        input=text,
        model="text-embedding-ada-002"
    )

# Test case 2: Edge case - empty text
def test_get_embeddings_empty_text(mock_client, mock_connect_openai):
    """
    Test the get_embeddings function with an empty text.
    """
    text = ""
    with pytest.raises(ValueError):
        get_embeddings(text)
    mock_connect_openai.assert_called_once()
    client = mock_connect_openai.return_value
    client.embeddings.create.assert_called_once_with(
        input=text,
        model="text-embedding-ada-002"
    )

# Test case 3: Error handling - API request failure
@patch('Unknown.get_openai_client')
def test_get_embeddings_api_failure(get_openai_client_mock):
    """
    Test the get_embeddings function when the API request fails.
    """
    get_openai_client_mock.return_value.embeddings.create.side_effect = Exception("API request failed")
    with pytest.raises(Exception):
        get_embeddings("This is a text")
    get_openai_client_mock.assert_called_once()

# Test case 4: Special case - invalid model
def test_get_embeddings_invalid_model(mock_client, mock_connect_openai):
    """
    Test the get_embeddings function with an invalid model.
    """
    text = "This is a text"
    with pytest.raises(ValueError):
        get_embeddings(text, model="invalid-model")
    mock_connect_openai.assert_called_once()
    client = mock_connect_openai.return_value
    client.embeddings.create.assert_called_once_with(
        input=text,
        model="invalid-model"
    )
