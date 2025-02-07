# import colorsys
# import hashlib

# class ColorGenerator:
#     def __init__(self, scale='0-255'):
#         self.scale = scale
#         self.base_colors = [
#             (255, 0, 0),     # Red
#             (0, 255, 0),     # Green
#             (0, 0, 255),     # Blue
#             (209, 134, 0),   # Orange
#             (0, 255, 255),   # Cyan
#             (255, 0, 255),   # Magenta
#         ]
#         self.generated_colors = {}  # Store colors assigned to seeds

#     def _string_to_hue(self, seed):
#         # Convert string to a hash and extract a consistent hue value
#         hash_object = hashlib.md5(seed.encode())
#         hash_int = int(hash_object.hexdigest(), 16)
#         return (hash_int % 360) / 360  # Map to range [0, 1]

#     def _generate_color(self, n):
#         if n < len(self.base_colors):
#             return self.base_colors[n]
#         else:
#             # For indices beyond base colors, generate colors using hue rotation
#             hue = (n - len(self.base_colors)) * 0.618033988749895 % 1
#             saturation = 0.8
#             value = 0.95
#             rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
#             return tuple(int(255 * x) for x in rgb_float)

#     def get_color(self, seed):
#         if seed in self.generated_colors:
#             return self.generated_colors[seed]

#         if len(self.generated_colors) < len(self.base_colors):
#             color = self.base_colors[len(self.generated_colors)]
#         else:
#             hue = self._string_to_hue(seed)
#             saturation = 0.8
#             value = 0.95
#             rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
#             color = tuple(int(255 * x) for x in rgb_float)

#         # Adjust the format based on the scale
#         if self.scale == '0-1':
#             color = tuple(c / 255 for c in color)
#         elif self.scale != '0-255':
#             raise ValueError("Invalid scale argument. Use '0-1' or '0-255'.")

#         # Store the generated color for the seed
#         self.generated_colors[seed] = color
#         return color


import colorsys
import hashlib


class ColorGenerator:
    def __init__(self, scale='matplotlib'):
        """
        Initialize the ColorGenerator.

        Parameters:
        - scale (str): Determines the color format. Options:
            - '0-255': RGB values in the range 0-255.
            - '0-1': Normalized RGB values in the range 0-1.
            - 'matplotlib': Hexadecimal color strings for Matplotlib compatibility.
        """
        self.scale = scale
        self.base_colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (209, 134, 0),   # Orange
            (0, 255, 255),   # Cyan
            (255, 0, 255),   # Magenta
        ]
        self.generated_colors = {}  # Store colors assigned to seeds
        self.used_hues = []         # Keep track of hues to ensure distinctiveness

    def _string_to_hue(self, seed):
        """Convert a string to a hue value in the range [0, 1]."""
        hash_object = hashlib.md5(seed.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        return (hash_int % 360) / 360  # Map to a circular hue [0, 1]

    def _generate_color(self, n):
        """Generate a new color based on the golden ratio."""
        if n < len(self.base_colors):
            return self.base_colors[n]
        else:
            hue = (n - len(self.base_colors)) * 0.618033988749895 % 1  # Golden ratio spacing
            saturation = 0.8  # Vibrant colors
            value = 0.95      # Bright colors
            rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
            return tuple(int(255 * x) for x in rgb_float)

    def _convert_to_matplotlib_format(self, color):
        """Convert an RGB color to a Matplotlib-compatible format."""
        if self.scale == 'matplotlib':
            # Convert to hexadecimal format
            return "#{:02x}{:02x}{:02x}".format(*color)
        elif self.scale == '0-1':
            # Normalize to [0, 1]
            return tuple(c / 255 for c in color)
        elif self.scale == '0-255':
            return color
        else:
            raise ValueError("Invalid scale argument. Use '0-1', '0-255', or 'matplotlib'.")

    def get_color(self, seed):
        """
        Get a color for the given seed. If the seed is new, generate a distinct color.

        Parameters:
        - seed (str): A unique identifier for the color.

        Returns:
        - Color in the format specified by `self.scale`.
        """
        seed = str(seed)
        if seed in self.generated_colors:
            return self.generated_colors[seed]

        if len(self.generated_colors) < len(self.base_colors):
            color = self.base_colors[len(self.generated_colors)]
        else:
            hue = self._string_to_hue(seed)
            saturation = 0.8
            value = 0.95
            rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
            color = tuple(int(255 * x) for x in rgb_float)

        # Convert to the desired format and store the result
        formatted_color = self._convert_to_matplotlib_format(color)
        self.generated_colors[seed] = formatted_color
        return formatted_color
