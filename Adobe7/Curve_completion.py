import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

def process_curve(points, mode='connected'):
    """
    Process and complete the curve based on the specified mode.
    
    Parameters:
        points (np.ndarray): Array of input points.
        mode (str): Mode of occlusion ('connected' or 'disconnected').
        
    Returns:
        np.ndarray: Array of completed points.
    """
    if mode == 'connected':
        return linear_interpolation(points)
    elif mode == 'disconnected':
        return connect_endpoints(points)
    else:
        raise ValueError(f"Invalid mode: {mode}")

def linear_interpolation(points):
    """
    Complete the curve using linear interpolation.
    
    Parameters:
        points (np.ndarray): Array of points for interpolation.
        
    Returns:
        np.ndarray: Array of points after interpolation.
    """
    x_vals, y_vals = points[:, 0], points[:, 1]
    interpolator = interp1d(x_vals, y_vals, kind='linear')
    x_new = np.linspace(x_vals[0], x_vals[-1], 500)
    y_new = interpolator(x_new)
    return np.column_stack((x_new, y_new))

def connect_endpoints(points):
    """
    Connect endpoints of fragmented curves using a linear regression model.
    
    Parameters:
        points (np.ndarray): Array of points to be connected.
        
    Returns:
        np.ndarray: Array of points including the connected line.
    """
    if len(points) < 2:
        return points

    x_start, x_end = points[0][0], points[-1][0]
    reg_model = LinearRegression()
    reg_model.fit(points[:, [0]], points[:, 1])
    
    x_range = np.linspace(x_start, x_end, 500)
    y_range = reg_model.predict(x_range[:, np.newaxis])
    
    return np.vstack((points, np.column_stack((x_range, y_range))))

# Example usage in __1.py
if __name__ == "__main__":
    from utils import read_csv, plot, polylines2svg
    from Regulariztion import fit_line, fit_circle
    from Symmetry_Detection import detect_symmetry
    from Curve_completion import process_curve

    csv_input = 'problems/problems/isolated.csv'
    svg_output = 'problems/problems/output.svg'
    
    curves_data = read_csv(csv_input)
    for curve in curves_data:
        for pts in curve:
            line_params = fit_line(pts)
            print(f'Line Parameters: slope = {line_params[0]}, intercept = {line_params[1]}')
            
            circle_params = fit_circle(pts)
            print(f'Circle Parameters: center = ({circle_params[0]}, {circle_params[1]}), radius = {circle_params[2]}')
            
            symmetry_info = detect_symmetry(pts)
            print(f'Symmetry Information: {symmetry_info}')
            
            completed_pts = process_curve(pts, mode='connected')
            print(f'Completed Points: {completed_pts}')
            
            plot(curves_data)
            polylines2svg(curves_data, svg_output)
