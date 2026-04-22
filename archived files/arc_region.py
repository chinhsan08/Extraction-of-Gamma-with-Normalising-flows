import numpy as np
from matplotlib.path import Path

class ArcRegionDetector:
    """
    Detector for arc-bounded region to check if points fall inside.
    """
    
    def __init__(self, left_start=(0.28, 0.45), left_end=(0.375, 0.58), 
                 shift=0.04, n_points=100, curvature=0.15):
        """
        Initialize the arc region boundary.
        
        Parameters:
        -----------
        left_start : tuple
            Start point of left arc (x, y)
        left_end : tuple
            End point of left arc (x, y)
        shift : float
            Horizontal shift for right arc (region width)
        n_points : int
            Number of points per arc (higher = smoother boundary)
        curvature : float
            Arc curvature (0 = straight, higher = more curved)
        """
        self.left_start = left_start
        self.left_end = left_end
        self.shift = shift
        self.curvature = curvature
        
        # Create boundary polygon
        self.boundary = self._create_boundary(left_start, left_end, shift, n_points, curvature)
        self.path = Path(self.boundary)
    
    def _create_arc_points(self, start, end, n_points, curvature):
        """Create points along an arc using quadratic Bezier curve."""
        x1, y1 = start
        x2, y2 = end
        
        t = np.linspace(0, 1, n_points)
        
        # Midpoint and perpendicular direction
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        # Control point offset perpendicular to the line
        perp_x = -dy / length
        perp_y = dx / length
        control_offset = length * curvature
        ctrl_x = mid_x + perp_x * control_offset
        ctrl_y = mid_y + perp_y * control_offset
        
        # Quadratic Bezier curve
        arc_x = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
        arc_y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
        
        return np.column_stack([arc_x, arc_y])
    
    def _create_boundary(self, left_start, left_end, shift, n_points, curvature):
        """Create closed boundary polygon from two parallel arcs."""
        # Left arc
        left_arc = self._create_arc_points(left_start, left_end, n_points, curvature)
        
        # Right arc (shifted)
        right_start = (left_start[0] + shift, left_start[1])
        right_end = (left_end[0] + shift, left_end[1])
        right_arc = self._create_arc_points(right_start, right_end, n_points, curvature)
        
        # Combine to form closed polygon
        boundary = np.vstack([
            left_arc,
            right_arc[::-1],  # Reverse right arc
            left_arc[0:1]     # Close the polygon
        ])
        
        return boundary
    
    def is_inside(self, x, y):
        """
        Check if a single point (x, y) is inside the region.
        
        Parameters:
        -----------
        x : float
            X coordinate
        y : float
            Y coordinate
        
        Returns:
        --------
        bool : True if inside, False otherwise
        """
        return self.path.contains_point((x, y))
    
    def check_points(self, points):
        """
        Check multiple points at once.
        
        Parameters:
        -----------
        points : array-like of shape (N, 2)
            Array of (x, y) coordinates
        
        Returns:
        --------
        numpy.ndarray : Boolean array of length N
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return self.path.contains_points(points)