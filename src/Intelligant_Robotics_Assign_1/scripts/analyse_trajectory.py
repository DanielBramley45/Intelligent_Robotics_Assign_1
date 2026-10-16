#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.expanduser(
    '~/ros2_jazzy_ws/src/Intelligant_Robotics_Assign_1/results'
)

ODOM_FILE = os.path.join(
    RESULTS_DIR, 'odometry.csv'
)

GROUND_TRUTH_FILE = os.path.join(
    RESULTS_DIR, 'ground_truth.csv'
)


def load_data():
    odom = pd.read_csv(ODOM_FILE)
    truth = pd.read_csv(GROUND_TRUTH_FILE)

    if odom.empty or truth.empty:
        raise ValueError(
            'One or both CSV files contain no recorded samples.'
        )

    for name, data in [
        ('odometry', odom),
        ('ground truth', truth)
    ]:
        if not np.isfinite(
            data[['time', 'x', 'y', 'z']].to_numpy()
        ).all():
            raise ValueError(
                f'{name} contains invalid coordinate or time values.'
            )

    # Sort by timestamp and remove duplicate timestamps.
    odom = (
        odom.sort_values('time')
        .drop_duplicates(subset='time')
        .reset_index(drop=True)
    )

    truth = (
        truth.sort_values('time')
        .drop_duplicates(subset='time')
        .reset_index(drop=True)
    )

    # Use only the time interval covered by both streams.
    start_time = max(
        odom['time'].min(),
        truth['time'].min()
    )

    end_time = min(
        odom['time'].max(),
        truth['time'].max()
    )

    if end_time <= start_time:
        raise ValueError(
            'The odometry and ground-truth timestamps do not overlap. '
            'Check that both use simulation time.'
        )

    odom = odom[
        (odom['time'] >= start_time)
        & (odom['time'] <= end_time)
    ].copy()

    truth = truth[
        (truth['time'] >= start_time)
        & (truth['time'] <= end_time)
    ].copy()

    if len(odom) < 2 or len(truth) < 2:
        raise ValueError(
            'Not enough overlapping samples to compare trajectories.'
        )

    return odom, truth


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    odom, truth = load_data()

    # Interpolate ground-truth coordinates at each odometry time.
    aligned = pd.DataFrame({
        'time': odom['time'].to_numpy(),
        'odom_x': odom['x'].to_numpy(),
        'odom_y': odom['y'].to_numpy(),
        'true_x': np.interp(
            odom['time'], truth['time'], truth['x']
        ),
        'true_y': np.interp(
            odom['time'], truth['time'], truth['y']
        )
    })

    # Calculate position errors.
    aligned['error_x'] = (
        aligned['odom_x'] - aligned['true_x']
    )

    aligned['error_y'] = (
        aligned['odom_y'] - aligned['true_y']
    )

    aligned['euclidean_error'] = np.sqrt(
        aligned['error_x'] ** 2
        + aligned['error_y'] ** 2
    )

    # Save all matched coordinates and errors.
    error_file = os.path.join(
        RESULTS_DIR, 'trajectory_error.csv'
    )

    aligned.to_csv(error_file, index=False)

    # Summary statistics.
    final = aligned.iloc[-1]

    final_error = final['euclidean_error']
    mean_error = aligned['euclidean_error'].mean()
    max_error = aligned['euclidean_error'].max()
    rmse = np.sqrt(
        np.mean(aligned['euclidean_error'] ** 2)
    )

    print('\nTRAJECTORY ERROR SUMMARY')
    print('------------------------')
    print(f'Compared samples: {len(aligned)}')
    print(f'Comparison start: {aligned["time"].iloc[0]:.3f} s')
    print(f'Comparison end:   {aligned["time"].iloc[-1]:.3f} s')

    print('\nFinal estimated position:')
    print(
        f'  x = {final["odom_x"]:.4f} m, '
        f'y = {final["odom_y"]:.4f} m'
    )

    print('\nFinal ground-truth position:')
    print(
        f'  x = {final["true_x"]:.4f} m, '
        f'y = {final["true_y"]:.4f} m'
    )

    print('\nPosition error:')
    print(f'  Final error: {final_error:.4f} m')
    print(f'  Mean error:  {mean_error:.4f} m')
    print(f'  Maximum:     {max_error:.4f} m')
    print(f'  RMSE:        {rmse:.4f} m')

    print(f'\nMatched data saved to: {error_file}')

    # Plot 1: Estimated and true paths.
    plt.figure(figsize=(8, 7))
    plt.plot(
        aligned['true_x'],
        aligned['true_y'],
        label='Ground truth'
    )
    plt.plot(
        aligned['odom_x'],
        aligned['odom_y'],
        label='Estimated odometry'
    )
    plt.scatter(
        aligned['true_x'].iloc[0],
        aligned['true_y'].iloc[0],
        marker='o',
        label='Ground-truth start'
    )
    plt.scatter(
        aligned['true_x'].iloc[-1],
        aligned['true_y'].iloc[-1],
        marker='x',
        label='Ground-truth end'
    )

    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('Estimated Odometry vs Ground Truth')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, 'trajectory_comparison.png'),
        dpi=200
    )

    # Plot 2: Position error over time.
    plt.figure(figsize=(10, 5))
    plt.plot(
        aligned['time'],
        aligned['euclidean_error']
    )

    plt.xlabel('Simulation time (s)')
    plt.ylabel('Euclidean position error (m)')
    plt.title('Position Error Over Time')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, 'position_error.png'),
        dpi=200
    )

    # Plot 3: X and Y errors over time.
    plt.figure(figsize=(10, 5))
    plt.plot(
        aligned['time'],
        aligned['error_x'],
        label='X error'
    )
    plt.plot(
        aligned['time'],
        aligned['error_y'],
        label='Y error'
    )

    plt.xlabel('Simulation time (s)')
    plt.ylabel('Coordinate error (m)')
    plt.title('X and Y Position Errors')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, 'coordinate_errors.png'),
        dpi=200
    )

    print('\nPlots saved:')
    print('  trajectory_comparison.png')
    print('  position_error.png')
    print('  coordinate_errors.png')

    plt.show()


if __name__ == '__main__':
    main()
