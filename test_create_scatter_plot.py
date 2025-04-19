# tests/test_scatter_plot.py

import pytest
import matplotlib.pyplot as plt
import numpy as np
from Unknown import create_scatter_plot  # Replace Unknown with the actual module name
import os

# Fixture to create a temporary directory for plots
@pytest.fixture(autouse=True)
def setup_plots_dir(tmp_path):
    """Create a temporary directory for plots."""
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    yield plots_dir
    # Clean up the temporary directory
    for file in plots_dir.iterdir():
        file.unlink()

# Test case 1: Normal functionality with typical inputs
def test_create_scatter_plot_normal_usage(setup_plots_dir):
    """Test the function with typical inputs."""
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 6, 8, 10])   
    title = "Scatter Plot Example"
    filename = "scatter_plot.png"
    expected_path = os.path.join(setup_plots_dir, filename)

    # Call the function under test
    path = create_scatter_plot(x_data, y_data, title, filename)

    # Check if the plot was saved to the correct location
    assert path == expected_path

    # Check if the plot exists
    assert os.path.exists(expected_path)

    # Check if the plot has the correct title
    plt.imshow(plt.imread(expected_path))
    plt.close()
    assert plt.title(title) == title

# Test case 2: Edge case - empty input arrays
def test_create_scatter_plot_empty_arrays(setup_plots_dir):
    """Test the function with empty input arrays."""
    x_data = np.array([])
    y_data = np.array([])
    title = "Scatter Plot Example"
    filename = "scatter_plot.png"
    expected_path = os.path.join(setup_plots_dir, filename)

    # Call the function under test
    path = create_scatter_plot(x_data, y_data, title, filename)

    # Check if the plot was saved to the correct location
    assert path == expected_path

    # Check if the plot exists
    assert os.path.exists(expected_path)

# Test case 3: Error handling - invalid input types
def test_create_scatter_plot_invalid_input_types(setup_plots_dir):
    """Test the function with invalid input types."""
    x_data = "not an array"
    y_data = 123
    title = "Scatter Plot Example"
    filename = "scatter_plot.png"

    # Call the function under test and expect an error
    with pytest.raises(TypeError):
        create_scatter_plot(x_data, y_data, title, filename)

# Test case 4: Error handling - missing required arguments
def test_create_scatter_plot_missing_required_args(setup_plots_dir):
    """Test the function with missing required arguments."""
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 6, 8, 10])
    title = "Scatter Plot Example"
    filename = "scatter_plot.png"

    # Call the function under test and expect an error
    with pytest.raises(TypeError):
        create_scatter_plot(x_data, y_data, title)

# Test case 5: Error handling - invalid filename
def test_create_scatter_plot_invalid_filename(setup_plots_dir):
    """Test the function with an invalid filename."""
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 6, 8, 10])
    title = "Scatter Plot Example"
    filename = "invalid filename"

    # Call the function under test and expect an error
    with pytest.raises(TypeError):
        create_scatter_plot(x_data, y_data, title, filename)
