"""
app.py
goal: A helper chat application for ADT (Analog Design Toolkit), by MasterMicro, to assist in plotting MOSFET data in interactive plots.
author: Abdelrahman Sabry
date: 2024-7-15
version: 1.0
"""

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import re

from generate_csv import generate_mosfet_data
# --- Plotting Functions ---

def create_plotly_scatter(data_frame, x_column_name, y_column_name, color_by_column=None, color_map_scheme='viridis'):
    """
    Creates an interactive scatter plot using Plotly Express.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing the data to plot.
        x_column_name (str): The name of the column to be used for the X-axis.
        y_column_name (str): The name of the column to be used for the Y-axis.
        color_by_column (str, optional): The name of the column to use for coloring
                                         the scatter points. Defaults to None.
        color_map_scheme (str, optional): The Plotly color scheme to use if
                                          color_by_column is numeric. Defaults to 'viridis'.

    Returns:
        plotly.graph_objects.Figure: A Plotly Figure object representing the scatter plot,
                                     or None if there's an error (e.g., missing columns).
    """
    if data_frame.empty:
        st.error("No data available for plotting")
        return None
    if x_column_name not in data_frame.columns:
        st.error(f"X-axis column '{x_column_name}' not found. Available columns: {list(data_frame.columns)}")
        return None
    if y_column_name not in data_frame.columns:
        st.error(f"Y-axis column '{y_column_name}' not found. Available columns: {list(data_frame.columns)}")
        return None
    if color_by_column and color_by_column not in data_frame.columns:
        st.error(f"Hue variable '{color_by_column}' not found. Available columns: {list(data_frame.columns)}")
        return None

    plotting_data_frame = data_frame.copy()
    plot_title = f"{y_column_name} vs {x_column_name}"
    if color_by_column:
        plot_title += f" (colored by {color_by_column})"

    # Create scatter plot using Plotly Express
    scatter_figure = px.scatter(
        plotting_data_frame,
        x=x_column_name,
        y=y_column_name,
        color=color_by_column,
        color_continuous_scale=color_map_scheme if color_by_column and pd.api.types.is_numeric_dtype(plotting_data_frame[color_by_column]) else None,
        title=plot_title,
        labels={x_column_name: x_column_name, y_column_name: y_column_name, color_by_column: color_by_column if color_by_column else None},
        hover_data=[color_by_column] if color_by_column else None
    )

    # Update plot layout for better aesthetics and readability
    scatter_figure.update_layout(
        width=800,
        height=600,
        xaxis_title=x_column_name,
        yaxis_title=y_column_name,
        showlegend=True if color_by_column else False
    )
    # Customize marker size and opacity
    scatter_figure.update_traces(marker=dict(size=8, opacity=0.7))
    return scatter_figure

# --- Data Operations Class ---

class DataOperationsHandler:
    """
    A class to handle operations on a DataFrame, including computing derived
    columns and applying conditional filtering.
    """
    def __init__(self, original_data_frame):
        """
        Initializes the DataOperationsHandler with a DataFrame.

        Args:
            original_data_frame (pd.DataFrame): The initial DataFrame to operate on.
        """
        self.original_data_frame = original_data_frame.copy()
        self.filtered_data_frame = original_data_frame.copy() # Stores the DataFrame after applying conditions

    def compute_derived_columns(self, x_expression, y_expression):
        """
        Computes new columns based on mathematical expressions for X and Y axes.
        If an expression is provided and it's not an existing column, a new
        derived column is added to the DataFrame.

        Args:
            x_expression (str): The expression for the X-axis (e.g., "gm/Id").
            y_expression (str): The expression for the Y-axis (e.g., "intrinsic_gain * 2").

        Returns:
            tuple: A tuple containing:
                   - pd.DataFrame: The DataFrame with potentially new derived columns.
                   - str: The name of the newly created X-axis column (or None if not created).
                   - str: The name of the newly created Y-axis column (or None if not created).
        """
        modified_data_frame = self.original_data_frame.copy()
        new_x_axis_column, new_y_axis_column = None, None

        # Helper lambda to check if a string is a mathematical expression
        is_expression = lambda expr: expr and any(op in expr for op in ['/', '*', '+', '-'])

        try:
            # Evaluate X-axis expression if it's a new derived column
            if is_expression(x_expression) and x_expression not in modified_data_frame.columns:
                new_x_axis_column = f"derived_{re.sub('[^0-9a-zA-Z_]', '_', x_expression)}"
                modified_data_frame[new_x_axis_column] = modified_data_frame.eval(x_expression)
        except Exception as e:
            st.warning(f"Could not evaluate x-axis expression '{x_expression}': {e}")

        try:
            # Evaluate Y-axis expression if it's a new derived column
            if is_expression(y_expression) and y_expression not in modified_data_frame.columns:
                new_y_axis_column = f"derived_{re.sub('[^0-9a-zA-Z_]', '_', y_expression)}"
                modified_data_frame[new_y_axis_column] = modified_data_frame.eval(y_expression)
        except Exception as e:
            st.warning(f"Could not evaluate y-axis expression '{y_expression}': {e}")

        return modified_data_frame, new_x_axis_column, new_y_axis_column

    def compute_conditional_df(self, input_data_frame, conditions_list):
        """
        Applies a list of conditions to filter the DataFrame.

        Args:
            input_data_frame (pd.DataFrame): The base DataFrame to apply conditions on.
            conditions_list (list): A list of condition strings (e.g., "Id > 1e-5").
        """
        self.filtered_data_frame = input_data_frame.copy()
        for single_condition in conditions_list:
            try:
                # Parse the condition string into LHS, operator, and RHS
                left_hand_side, comparison_operator, right_hand_side = parse_condition(single_condition)
                if left_hand_side not in self.filtered_data_frame.columns:
                    st.warning(f"Column '{left_hand_side}' not found. Skipping condition: {single_condition}")
                    continue
                # Convert RHS to float, handling scientific notation and units
                right_hand_side_value = float(parse_from_number(right_hand_side))

                # Apply the filter based on the operator
                if comparison_operator == ">":
                    self.filtered_data_frame = self.filtered_data_frame[self.filtered_data_frame[left_hand_side] > right_hand_side_value]
                elif comparison_operator == "<":
                    self.filtered_data_frame = self.filtered_data_frame[self.filtered_data_frame[left_hand_side] < right_hand_side_value]
                elif comparison_operator == "=":
                    self.filtered_data_frame = self.filtered_data_frame[self.filtered_data_frame[left_hand_side] == right_hand_side_value]
            except Exception as e:
                st.warning(f"Invalid condition '{single_condition}': {e}")

# --- Utility Functions ---

def parse_condition(input_condition_string):
    """
    Parses a single condition string into its left-hand side (column name),
    operator, and right-hand side (value).

    Args:
        input_condition_string (str): The condition string (e.g., "Vgs > 0.8").

    Returns:
        tuple: A tuple containing (column_name, operator, value_string).

    Raises:
        ValueError: If the condition string format is invalid.
    """
    # Regex pattern to capture column name, operator (>, <, =), and numeric value
    regex_pattern = r'(\w+)\s*([><=]+)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
    regex_match = re.match(regex_pattern, input_condition_string.strip())
    if regex_match:
        return regex_match.groups()
    raise ValueError(f"Invalid condition format: {input_condition_string}")

def parse_from_number(input_formatted_value):
    """
    Parses a string that might contain scientific notation or unit prefixes
    (e.g., "10u", "1.2m", "5k") into a float.

    Args:
        input_formatted_value (str): The string value to parse.

    Returns:
        float: The parsed numeric value.

    Raises:
        ValueError: If the input string cannot be parsed.
    """
    if pd.isna(input_formatted_value):
        return input_formatted_value
    input_formatted_value = str(input_formatted_value)
    # Dictionary mapping unit prefixes to their multipliers
    unit_multipliers = {"T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "": 1,
                        "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
    # Regex to capture the number and an optional unit prefix
    regex_match = re.match(r"([-+]?\d*\.?\d+(?:[eE][+-]?\d+)?)([TGMkmunpf]?)", input_formatted_value)
    if not regex_match:
        try:
            return float(input_formatted_value) # Try direct conversion if no match
        except ValueError:
            raise ValueError(f"Invalid formatted value: {input_formatted_value}")
    numeric_value = float(regex_match.group(1))
    unit_prefix = regex_match.group(2)
    return numeric_value * unit_multipliers[unit_prefix]

def parse_chat_input(chat_input_string, data_frame_columns):
    """
    Parses a natural language chat input string to extract plotting parameters
    like X-axis, Y-axis, hue variable, and filtering conditions.
    Supports case-insensitive column matching.

    Args:
        chat_input_string (str): The user's input string (e.g., "plot Id vs Vgs by L where Vgs > 0.5").
        data_frame_columns (list): A list of available column names in the DataFrame.

    Returns:
        dict: A dictionary containing parsed parameters:
              'x_axis', 'y_axis', 'conditions', 'hue_variable', 'colormap', 'success'.
    """
    parsed_parameters = {
        'x_axis': None, 'y_axis': None, 'conditions': [], 'hue_variable': None,
        'colormap': 'viridis', 'success': False
    }
    # Regex for valid column/expression characters
    expression_characters_regex = r'[a-zA-Z0-9_/\.\*+-]+'
    column_characters_regex = r'[a-zA-Z_][a-zA-Z0-9_]*'

    # Convert DataFrame columns to lowercase for case-insensitive matching
    lower_case_df_columns = [col.lower() for col in data_frame_columns]
    # Map lowercase column names back to their original case
    column_name_map = dict(zip(lower_case_df_columns, data_frame_columns))

    # Debug: print the input
    print(f"DEBUG: Parsing input: '{chat_input_string}'")

    # Check for "by" grouping (e.g., "plot X vs Y by Z")
    plot_by_regex_match = re.search(rf'(?:plot|scatter)\s+({expression_characters_regex})\s+vs\s+({expression_characters_regex})\s+by\s+({column_characters_regex})', chat_input_string.lower())
    if plot_by_regex_match:
        x_axis_input, y_axis_input, hue_variable_input = plot_by_regex_match.group(1), plot_by_regex_match.group(2), plot_by_regex_match.group(3)
        parsed_parameters['x_axis'] = column_name_map.get(x_axis_input.lower(), x_axis_input)
        parsed_parameters['y_axis'] = column_name_map.get(y_axis_input.lower(), y_axis_input)
        parsed_parameters['hue_variable'] = column_name_map.get(hue_variable_input.lower(), hue_variable_input)
        parsed_parameters['success'] = True
        print(f"DEBUG: Found 'by' grouping - x: {parsed_parameters['x_axis']}, y: {parsed_parameters['y_axis']}, hue: {parsed_parameters['hue_variable']}")
        # Return early if a complete plot command with 'by' is found
        return parsed_parameters

    # Check for "colored by" grouping (e.g., "color by Z") - can be combined with basic plot
    colored_by_regex_match = re.search(rf'color(?:ed)?\s+by\s+({column_characters_regex})', chat_input_string.lower())
    if colored_by_regex_match:
        hue_variable_input = colored_by_regex_match.group(1)
        parsed_parameters['hue_variable'] = column_name_map.get(hue_variable_input.lower(), hue_variable_input)
        print(f"DEBUG: Found 'colored by' grouping - hue: {parsed_parameters['hue_variable']}")

    # Check for basic plot pattern (e.g., "plot X vs Y")
    plot_regex_match = re.search(rf'(?:plot|scatter)\s+({expression_characters_regex})\s+vs\s+({expression_characters_regex})', chat_input_string.lower())
    if plot_regex_match:
        x_axis_input, y_axis_input = plot_regex_match.group(1), plot_regex_match.group(2)
        parsed_parameters['x_axis'] = column_name_map.get(x_axis_input.lower(), x_axis_input)
        parsed_parameters['y_axis'] = column_name_map.get(y_axis_input.lower(), y_axis_input)
        parsed_parameters['success'] = True
        print(f"DEBUG: Found basic plot - x: {parsed_parameters['x_axis']}, y: {parsed_parameters['y_axis']}")

    # Parse conditions (e.g., "where Vgs > 0.5")
    condition_regex_pattern = r'(?:where\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*([><=]+)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?\w?)'
    condition_matches = re.findall(condition_regex_pattern, chat_input_string.lower())
    for single_match in condition_matches:
        # Avoid parsing 'vs' or 'by' as column names for conditions
        if single_match[0] not in ['vs', 'by']:
            formatted_condition = f"{column_name_map.get(single_match[0].lower(), single_match[0])} {single_match[1]} {single_match[2]}"
            if formatted_condition not in parsed_parameters['conditions']:
                parsed_parameters['conditions'].append(formatted_condition)

    print(f"DEBUG: Final parsed params: {parsed_parameters}")
    return parsed_parameters

# --- Streamlit Application Functions ---

def main():
    """
    Main function for the Streamlit ADT Assistant application.
    Sets up the page, handles data generation, user input, plotting,
    and displays results.
    """
    st.set_page_config(page_title="ADT Assistant", page_icon="📊", layout="wide")
    initialize_session_state()

    st.title("📊 ADT Assistant: MOSFET Data Plotter")
    st.markdown("Chat to create scatter plots from MOSFET data!")

    # Generate and load data if not already in session state
    if 'data_frame' not in st.session_state or st.session_state.data_frame is None:
        current_data_frame = generate_mosfet_data()
        st.session_state.data_frame = current_data_frame
        st.session_state.data_operations_object = DataOperationsHandler(current_data_frame)
        st.success(f"✅ Generated MOSFET data with {len(current_data_frame)} points!")

    # Sidebar for dataset information
    with st.sidebar:
        st.header("Dataset Info")
        if st.session_state.data_frame is not None:
            st.metric("Rows", len(st.session_state.data_frame))
            st.metric("Columns", len(st.session_state.data_frame.columns))
            with st.expander("Available Columns"):
                st.write(list(st.session_state.data_frame.columns))
            with st.expander("Sample Data"):
                st.dataframe(st.session_state.data_frame.head())

    # Display chat interface and previous messages
    st.header("Chat Interface")
    for i, chat_message in enumerate(st.session_state.messages):
        with st.chat_message(chat_message["role"]):
            st.markdown(chat_message["content"])
            # If a plot was generated for this message, display it
            if "plot" in chat_message and chat_message["plot"] is not None:
                st.plotly_chart(chat_message["plot"], use_container_width=True, key=f"plot_{i}")

    # Handle new user input from the chat input box
    if user_prompt := st.chat_input("e.g., 'plot gmoverid vs intrinsic_gain by L'"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        try:
            # Parse the user's chat input to extract plot parameters
            parsed_plot_parameters = parse_chat_input(user_prompt, st.session_state.data_frame.columns)

            if not parsed_plot_parameters['success']:
                # If parsing failed, inform the user about the correct command format
                error_message = "Please use a command like 'plot gmoverid vs intrinsic_gain by L'"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.markdown(error_message)
            else:
                x_axis_expression = parsed_plot_parameters['x_axis']
                y_axis_expression = parsed_plot_parameters['y_axis']

                # Compute derived columns if expressions are provided
                data_with_derived_columns, new_x_column_name, new_y_column_name = \
                    st.session_state.data_operations_object.compute_derived_columns(x_axis_expression, y_axis_expression)

                # Update the x/y axis names if new derived columns were created
                if new_x_column_name:
                    parsed_plot_parameters['x_axis'] = new_x_column_name
                if new_y_column_name:
                    parsed_plot_parameters['y_axis'] = new_y_column_name

                # Apply conditional filtering to the data
                st.session_state.data_operations_object.compute_conditional_df(data_with_derived_columns, parsed_plot_parameters['conditions'])

                # Create the Plotly scatter plot
                generated_figure = create_plotly_scatter(
                    st.session_state.data_operations_object.filtered_data_frame,
                    parsed_plot_parameters['x_axis'], parsed_plot_parameters['y_axis'],
                    color_by_column=parsed_plot_parameters['hue_variable'],
                    color_map_scheme=parsed_plot_parameters['colormap']
                )

                if generated_figure:
                    # Construct the assistant's response message
                    assistant_response = f"Got it!! I am going to plot **{y_axis_expression}** vs **{x_axis_expression}**."
                    if parsed_plot_parameters['hue_variable']:
                        assistant_response += f" Colored by **{parsed_plot_parameters['hue_variable']}**."
                    if parsed_plot_parameters['conditions']:
                        assistant_response += f" Filtered by: {', '.join(parsed_plot_parameters['conditions'])}"

                    # Add the response and plot to the session messages
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response, "plot": generated_figure})
                    with st.chat_message("assistant"):
                        st.markdown(assistant_response)
                        st.plotly_chart(generated_figure, use_container_width=True)
                else:
                    # Handle cases where plot creation failed
                    error_message = f"Could not create plot. Check column names or expressions. Available columns: {list(st.session_state.data_frame.columns)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
        except Exception as e:
            # Catch any unexpected errors during processing
            error_message = f"An unexpected error occurred: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.markdown(error_message)

def initialize_session_state():
    """
    Initializes Streamlit session state variables to ensure they exist
    before being accessed. This prevents KeyError on first run.
    """
    if 'data_frame' not in st.session_state:
        st.session_state.data_frame = None
    if 'data_operations_object' not in st.session_state:
        st.session_state.data_operations_object = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

# Entry point for the Streamlit application
if __name__ == "__main__":
    main()
