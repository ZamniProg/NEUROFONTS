# Project Overview: Handwritten Text Recognition and Generation

## **Idea:**

We are developing a project focused on the following key tasks:

### 1. **Text Recognition in Images, Adapted to the Specific User.**
The system will analyze images containing text, recognize the characters and their relationships (e.g., connections between letters), and account for the individual features of the user's handwriting to improve recognition accuracy. The collected data (such as letter positions, sizes, and tilt angles) will be saved in a convenient format (e.g., JSON) and added to fonts.

This will work with an OCR system like [TrOCR](https://github.com/microsoft/unilm/tree/master/trOCR), later retrained to recognize Russian symbols and letter bounding boxes. Once the bounding boxes are recognized, the model will generate a font using GAN or VAE neural networks.

### 2. **Updated Text Recognition for Summary Generation.**
To fully complete the application, a summary generation feature will be added. This allows the AI to:
- Collect data such as the size of words (e.g., headers, labels, main text), position on the page (left, right, center, width alignment), and text style (bold, italic).
- Recognize structures like graphs and tables and save this data to format it into text in a text editor. If the image doesn't contain a graph or table, the AI will crop the image and insert it into the summary.

We plan to use Faster R-CNN and image formatting algorithms for this feature.

### 3. **Text Generation in Images with Realism.**
The user will input text, which the system will generate in their handwriting style and overlay on an image (e.g., a photo of a sheet of paper), with realistic details like:
- Slightly uneven lines
- Varying letter sizes
- Tilts and other features to make the result appear natural.

The project will combine these technologies to:
- Improve text recognition for a specific user by adapting to their handwriting.
- Assist students and other users in creating realistic handwritten text (e.g., for academic assignments).

GAN-based systems will be used to generate images with the user's font.

## **Task Analysis and Data Setup:**

### 1. **Text Recognition from Image:**
- **Dataset:** A dataset containing Russian text images with metadata in JSON format, including bounding boxes for words and translations.
- **Requirements:**
  - Recognize letter hitboxes.
  - Recognize connections between letters.
  - Account for unique user font features.
  - Datasets for other languages (English, French, German, etc.) will be required.

### 2. **Summary Generator:**
- **Dataset:** Custom dataset needed as no such dataset exists for all languages.
- **Requirements:** Custom dataset creation.

### 3. **Text Generation:**
- **Dataset:**
  - Backgrounds for images.
  - User text examples for generation.
- **Requirements:**
  - Process backgrounds (e.g., a sheet of paper with text).
  - Overlay text with realistic features: irregular letters, rotations, textures.

## **System Architecture:**

### 1. **Network for Text Recognition:**
- **Approach:** Use an OCR network for text and font recognition. Incorporate Faster R-CNN to recognize the user's font style and adapt it for LaTeX and Word styles.
- **Functions:** 
  - Recognize text and metrics (letter shapes, angles, sizes, connections).
  - Generate metadata in JSON format (bounding boxes, text, connections).

### 2. **Network for Text Generation:**
- **Approach:** Use GANs (or Diffusion Models) for realistic text generation, adapting to the user's font style (including deviations).
- **Functions:** 
  - Mix textures and overlay text.
  - Create a realistic font style with deviations, shades, and non-proportional scaling.

### 3. **User-Friendly Text Editor:**
- **Goals:** Develop an application in C# or C++ that is compatible with Linux, macOS, Windows (and possibly Android/iOS in the future). The interface will combine features from Obsidian and Word, with day/night modes.

### **Updates:**
1. Add recognition for math symbols.
2. Update summary generation for teachers and students.
3. Include a network panel with an AI helper for users to find relevant info for creating summaries.
4. Add a PDF reader.
5. Improve image recognition (graphs, math objects).
6. Merge this project with the Math Helper project (e.g., math tasks, 3D engine, graphs builder).

## **Networks Integration:**
- **Goal:** Combine the results of the first network (recognition) as input to the second network (generation).
- **Data Formats:** JSON file with text characteristics, allowing users to modify data before generation.

## **Realization:**

### 1. **Recognize Network:**
- **Libraries:** PyTorch/TensorFlow for machine learning.
- **Functions:**
  - Symbols recognition (CRNN, Tesseract for base OCR).
  - Detect letter connections (contour analysis, clustering).

### 2. **Generation Network:**
- **Libraries:** PyTorch (GAN, StyleGAN), OpenCV for text imposition.
- **Functions:**
  - Text generation in user font.
  - Add realistic effects (rotations, noise).

### 3. **UI (User Interface):**
- Telegram bot with an API.
- Desktop and/or web application.

## **Project Timeline:**
- **10.01.2025, 3AM:**
  - Start of work, dataset download from *"ai-forever/school_notebooks_RU"*.
  - Added functions for downloading and loading datasets.
  - Fixed annotation upload issue.
  
- **11.01.2025, 4AM:**
  - Need to check how the neural network for image recognition works.
  - Write comments for every class and function.

- **11.01.2025, 9:30 PM:**
  - Added preprocess class, fixed bugs.

- **12.01.2025, 11:30 PM:**
  - Need to rework UserFriendly class using the COCO library.
  - Optimize everything for better performance.

- **13.01.2025, 8:42 PM:**
  - Pre-work is ready, need to focus on neural network.
  - Added COCO pre-processing and more useful docstrings.
  - Updated Preprocessing class.

- **14.01.2025, 6:40 PM:**
  - Check how updated functions work in PreProcess (NUMPY -> TORCH).
  - Fix errors with tensor sizes.

- **15.01.2025, 10:17 PM:**
  - Update errors in preprocessing classes for better output.
  - Rework model structure for faster RNN.

- **16.01.2025, 11:06 PM:**
  - Need to understand how Faster R-CNN works and how to construct it.

- **17.01.2025, 11:50 PM:**
  - Working on recomputing model parameters, input/output data, and layers.

- **19.01.2025, 10:12 PM:**
  - Need to decide between OCR or Faster R-CNN + CNN + letter splitter.
  - Rewrite RPN and ROI pooling.
  - Consider using OCR for text recognition and Faster R-CNN for image, table, graph detection.

## **References:**
1. [ResearchGate - Clothing Classification](https://www.researchgate.net/publication/358946012_Using_Deep_Learning_in_Real-Time_for_Clothing_Classification_with_Connected_Thermostats#pf7)
2. [Faster R-CNN Step by Step - Part I](https://dongjk.github.io/code/object+detection/keras/2018/05/21/Faster_R-CNN_step_by_step,_Part_I.html)
3. [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
4. [GeeksforGeeks - Faster R-CNN](https://www.geeksforgeeks.org/faster-r-cnn-ml/)
