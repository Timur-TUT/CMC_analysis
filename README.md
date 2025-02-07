# CMC Analysis Tool

This tool analyzes the characteristics of cracks in SAXS images of ceramic matrix composites (CMCs). It extracts numerical features from annotated cracks and visualizes their progression across different load conditions.

## Description

This tool processes annotated crack images to extract various shape and intensity features. It tracks crack development across sequential load images and categorizes cracks based on their involvement in fracture. The extracted features are automatically analyzed and visualized using multiple statistical and graphical methods.

### **Input Data**
- **Raw Images**: SAXS images of CMCs under different load conditions.
- **Annotation Images**: Binary masks indicating crack regions.
- **Fracture Mask Images**: Binary masks specifying fracture zones.

### **Features Extracted**
- **Shape Features**: Length, width, perimeter, area, curvature, compactness, roughness.
- **Intensity Features**: Median, maximum, variance of brightness values.

### **Visualization**
- **Box Plots & Violin Plots**: Distribution of shape and intensity features across load conditions.
- **Heatmaps**: Crack density and brightness variation.
- **Skeleton Maps**: Crack propagation patterns.
- **Bar Graphs**: Total crack area and number.

## Getting Started

### Dependencies
```
* numpy          | 1.26.4
* opencv-python  | 4.9.0.80
* matplotlib     | 3.9.0
* scipy          | 1.13.1
* h5py           | 3.11.0
* pandas         | 2.2.2
* scikit-image   | 0.24.0
```

### Installing
```
git clone https://github.com/Timur-TUT/CMC_analysis.git
```

### Executing program
```
python main.py
```

## Authors

T. A. Khudayberganov  
[@Timur](g212300905@edu.teu.ac.jp)

## Version History

* 1.0
    * Initial Release

