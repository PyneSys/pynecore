<!--
---
weight: 424
title: "color"
description: "Color functions and color constants"
icon: "palette"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["color", "library", "reference"]
---
-->

# color

The color namespace provides functions to work with colors and access their components. Colors are used throughout PyneCore for visual elements like lines, labels, plots, and overlays. You can create colors from RGB values, modify transparency, interpolate between colors using gradients, and extract individual color components.

## Quick Example

```python
from pynecore.lib import close, high, low, color, Color, ta, script

@script.indicator(title="Color Demo", overlay=True)
def main():
    # Use predefined colors
    plot_color: Color = color.green if close > ta.sma(close, 20) else color.red
    
    # Create custom color with transparency
    transparent_blue: Color = color.new(color.blue, transp=50)
    
    # Create color from RGB values
    custom_color: Color = color.rgb(255, 165, 0)
    
    # Create color gradient based on price position
    price_position: float = (close - low) / (high - low) if high != low else 0.5
    bar_color: Color = color.from_gradient(
        price_position, 0, 1, color.red, color.green
    )
    
    # Extract color components
    red_val: int = color.r(plot_color)
    green_val: int = color.g(plot_color)
    transp_val: int = color.t(plot_color)
```

## Functions

### r()

Retrieves the red component of a color.

| Parameter | Type | Description |
|-----------|------|-------------|
| color | Color | The color to extract from |

**Returns:** `int` — The red component value (0-255)

```python
red_comp: int = color.r(color.blue)  # 41
```

### g()

Retrieves the green component of a color.

| Parameter | Type | Description |
|-----------|------|-------------|
| color | Color | The color to extract from |

**Returns:** `int` — The green component value (0-255)

```python
green_comp: int = color.g(color.lime)  # 230
```

### b()

Retrieves the blue component of a color.

| Parameter | Type | Description |
|-----------|------|-------------|
| color | Color | The color to extract from |

**Returns:** `int` — The blue component value (0-255)

```python
blue_comp: int = color.b(color.aqua)  # 212
```

### t()

Retrieves the transparency component of a color.

| Parameter | Type | Description |
|-----------|------|-------------|
| color | Color | The color to extract from |

**Returns:** `int` — The transparency value (0-100, where 0 is fully opaque and 100 is fully transparent)

```python
transp: int = color.t(color.new(color.red, transp=75))  # 75
```

### new()

Creates a new color with modified transparency, optionally accepting a color object or hex string.

| Parameter | Type | Description |
|-----------|------|-------------|
| color | Color or str | A color object or hex string in "#RRGGBB" or "#RRGGBBAA" format |
| transp | float | Transparency percentage (0-100, default 0) |

**Returns:** `Color` — Color with the specified transparency

```python
semi_transparent: Color = color.new(color.blue, transp=50)
custom_hex: Color = color.new("#FF5733", transp=25)
```

### rgb()

Creates a new color from RGB component values with optional transparency.

| Parameter | Type | Description |
|-----------|------|-------------|
| r | int | Red value (0-255) |
| g | int | Green value (0-255) |
| b | int | Blue value (0-255) |
| transp | float | Transparency percentage (0-100, default 0) |

**Returns:** `Color` — Color with the specified RGB values and transparency

```python
orange: Color = color.rgb(255, 165, 0)
translucent_purple: Color = color.rgb(156, 39, 176, transp=50)
```

### from_gradient()

Interpolates a color from a gradient based on a value's position within a range. Values outside the range are clamped to the corresponding endpoint colors.

| Parameter | Type | Description |
|-----------|------|-------------|
| value | int or float | The value to map to the gradient |
| bottom_value | int or float | The lower bound of the range |
| top_value | int or float | The upper bound of the range |
| bottom_color | Color | The color at the bottom of the range |
| top_color | Color | The color at the top of the range |

**Returns:** `Color` — Interpolated color based on value's position in the range

```python
bar_color: Color = color.from_gradient(close, low, high, color.red, color.green)
```

## Constants

| Name | Hex Value | RGB |
|------|-----------|-----|
| aqua | #00BCD4 | (0, 188, 212) |
| black | #363A45 | (54, 58, 69) |
| blue | #2962FF | (41, 98, 255) |
| fuchsia | #E040FB | (224, 64, 251) |
| gray | #787B86 | (120, 123, 134) |
| green | #4CAF50 | (76, 175, 80) |
| lime | #00E676 | (0, 230, 118) |
| maroon | #880E4F | (136, 14, 79) |
| navy | #311B92 | (49, 27, 146) |
| olive | #808000 | (128, 128, 0) |
| orange | #FF9800 | (255, 152, 0) |
| purple | #9C27B0 | (156, 39, 176) |
| red | #F23645 | (242, 54, 69) |
| silver | #B2B5BE | (178, 181, 190) |
| teal | #089981 | (8, 153, 129) |
| white | #FFFFFF | (255, 255, 255) |
| yellow | #FDD835 | (253, 216, 53) |

## Compatibility

All `color` functions and constants are fully supported in PyneCore. The namespace provides complete compatibility with Pine Script v6 color operations.