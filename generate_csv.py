"""
generate_csv.py
goal: Generate synthetic MOSFET device characteristic data for simulation and analysis.
author: Abdelrahman Sabry
date: 2024-7-15
version: 1.0
"""
import pandas as pd
import numpy as np

# --- Data Generation Functions ---

def generate_mosfet_data():
    """
    Generates synthetic MOSFET (Metal-Oxide-Semiconductor Field-Effect Transistor)
    device characteristic data for simulation and analysis.

    The data includes various parameters like gate-source voltage (Vgs),
    device area, drain current (Id), channel length (L), bulk-source voltage (VSB),
    transconductance (gm), output conductance (gds), channel width (W),
    output resistance (rout), transconductance over drain current (gm/Id),
    and intrinsic voltage gain.

    Returns:
        pd.DataFrame: A DataFrame containing the generated MOSFET data.
                      The DataFrame also includes derived columns like 'gm/gds'
                      and 'WoverL'.
    """
    np.random.seed(42)  # For reproducibility of data generation
    num_data_points_per_length = 150
    # Channel lengths in meters, representing different technology nodes
    channel_length_values_meters = [0.18e-6, 0.35e-6, 0.5e-6, 0.8e-6, 1.0e-6, 1.5e-6]
    total_data_points = num_data_points_per_length * len(channel_length_values_meters)

    # Generate Gate-Source Voltage (Vgs) with a triangular distribution, peaking around 0.85V
    gate_source_voltage = 0.3 + (1.4 - 0.3) * np.abs(np.random.random(total_data_points) - 0.5) * 2

    # Generate device area (W*L) with a triangular-like distribution, clipped to realistic range
    base_area_square_meters = np.random.uniform(0, 100e-12, total_data_points)
    device_area_square_meters = np.abs(base_area_square_meters - 50e-12) + np.random.uniform(0, 20e-12, total_data_points)
    device_area_square_meters = np.clip(device_area_square_meters, 0, 100e-12)

    # Generate Drain Current (Id) with a smooth gradient based on area, with added noise
    drain_current = np.interp(device_area_square_meters, [0, 50e-12, 100e-12], [1e-6, 1e-4, 1e-6]) + np.random.uniform(-2e-7, 2e-7, total_data_points)
    drain_current = np.clip(drain_current, 1e-6, 1e-4)

    # Generate Channel Length (L) by randomly selecting from predefined discrete values
    channel_length = np.random.choice(channel_length_values_meters, total_data_points)

    # Generate other MOSFET parameters with realistic ranges and relationships
    bulk_source_voltage = np.random.uniform(0, 0.3, total_data_points)
    transconductance = drain_current * np.random.uniform(10, 25, total_data_points) / 1e3  # gm in Siemens (S)
    output_conductance = transconductance * np.random.uniform(0.01, 0.1, total_data_points)  # gds in Siemens (S)
    channel_width = device_area_square_meters / channel_length  # Calculate W from area and L
    channel_width = np.clip(channel_width, 0.2e-6, 50e-6)  # Constrain W to realistic range
    output_resistance = 1 / output_conductance
    transconductance_over_drain_current = transconductance / drain_current
    intrinsic_voltage_gain = transconductance * output_resistance

    # Create a Pandas DataFrame from the generated data
    mosfet_data_frame = pd.DataFrame({
        "Vgs": gate_source_voltage,
        "area": device_area_square_meters,
        "Id": drain_current,
        "L": channel_length,
        "VSB": bulk_source_voltage,
        "gm": transconductance,
        "gds": output_conductance,
        "W": channel_width,
        "rout": output_resistance,
        "gmoverid": transconductance_over_drain_current,
        "intrinsic_gain": intrinsic_voltage_gain
    })

    # Add derived columns for further analysis
    mosfet_data_frame["gm/gds"] = mosfet_data_frame["gm"] / mosfet_data_frame["gds"]
    mosfet_data_frame["WoverL"] = mosfet_data_frame["W"] / mosfet_data_frame["L"]

    # Remove any NaN (Not a Number) or infinite values that might result from calculations
    mosfet_data_frame = mosfet_data_frame.replace([np.inf, -np.inf], np.nan).dropna()

    # Print a summary of the generated dataset to the console
    print(f"\nDataset Summary:")
    print(f"Total points: {len(mosfet_data_frame)}")
    print(f"L values: {mosfet_data_frame['L'].unique() * 1e6} um")
    print(f"Id range: {mosfet_data_frame['Id'].min():.2e} to {mosfet_data_frame['Id'].max():.2e} A")
    print(f"gmoverid range: {mosfet_data_frame['gmoverid'].min():.1f} to {mosfet_data_frame['gmoverid'].max():.1f} S/A")

    # Save the generated data to a CSV file
    mosfet_data_frame.to_csv("simplified_mosfet_data.csv", index=False)
    print("✅ Simplified MOSFET data saved to 'simplified_mosfet_data.csv'")
    return mosfet_data_frame