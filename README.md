# puneetdwmu_Team_Curvetopia

# CURVETOPIA: A Journey into the World of Curves

## Project Overview

Welcome to *Curvetopia*, where our mission is to bring order and beauty to the world of 2D curves. This project is a comprehensive exploration of identifying, regularizing, and beautifying various shapes in 2D Euclidean space, with a primary focus on closed curves. The end goal is to transform these curves into cubic Bezier representations using SVG format.

## Objectives

The project is divided into three main objectives:

### 1. Regularizing Curves
- *Goal:* To identify and regularize curves into standard geometric shapes.
- *Shapes Covered:*
  - *Straight Lines:* Identify and regularize linear paths.
  - *Circles and Ellipses:* Detect curves where points are equidistant from a center or have two focal points.
  - *Rectangles and Rounded Rectangles:* Differentiate between sharp and rounded cornered rectangles.
  - *Regular Polygons:* Identify polygons with equal sides and angles.
  - *Star Shapes:* Detect central points with multiple radial arms forming star-like structures.
- *Activity:* Test the algorithm with various hand-drawn shapes and doodles to ensure it can distinguish between regular and irregular shapes.

### 2. Exploring Symmetry in Curves
- *Goal:* To identify the presence of symmetry, particularly reflection symmetry, in closed shapes.
- *Symmetry Detection:* 
  - *Reflection Symmetry:* Check for lines of symmetry where the shape can be divided into mirrored halves.
- *Activity:* Identify symmetrical properties by transforming curves into point representations, and fit identical Bezier curves on symmetric points.

### 3. Completing Incomplete Curves
- *Goal:* To complete 2D curves that have gaps or partial holes due to occlusion removal.
- *Curve Completion Techniques:*
  - *Fully Contained Shapes:* For example, one circle completely inside another.
  - *Partially Contained Shapes:* For instance, half a circle occluded by another shape.
  - *Disconnected Shapes:* Examples include a circle fragmented by an intersecting rectangle.
- *Activity:* Develop algorithms to identify and naturally complete these incomplete curves, ensuring smoothness, regularity, and symmetry.
