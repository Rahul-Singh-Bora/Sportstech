# 🏆 Sports Performance Tracker

Real-time sports performance tracking system using MediaPipe for pose detection and analysis. Track angles, velocity, and posture for multiple sports!

## 🎯 Supported Sports

1. **Cricket Bowling** 🏏
   - Arm speed tracking
   - Elbow angle analysis (legal bowling check)
   - Posture feedback

2. **Basketball Shooting** 🏀
   - Wrist release speed
   - Shooting arm angle
   - Knee bend analysis
   - Form validation

3. **Yoga** 🧘
   - Balance metrics
   - Joint angle tracking
   - Hip stability analysis
   - Posture assessment

4. **Boxing** 🥊
   - Punch speed (both hands)
   - Arm extension angles
   - Power level detection

5. **Squats/Weightlifting** 🏋️
   - Knee angle tracking
   - Squat depth assessment (Deep/Parallel/Partial)
   - Hip movement analysis
   - Form symmetry check

## 📋 Requirements

- Python 3.8+
- Webcam

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Rahul-Singh-Bora/Sportstech.git
cd Sportstech
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

Run the program:
```bash
python vidio_track.py
```

1. Select your sport from the menu (1-5)
2. The camera will open with real-time tracking
3. Live metrics will be displayed on screen:
   - Angles in degrees
   - Velocities in pixels/second
   - Posture feedback
   - Form assessment
4. Press **ESC** to exit

## 🛠️ Technologies Used

- **OpenCV** - Video capture and display
- **MediaPipe** - Pose detection and landmark tracking
- **NumPy** - Mathematical calculations
- **Python** - Core programming

## 📊 Metrics Calculated

- **Angles**: Joint angles calculated using 3D coordinates
- **Velocity**: Real-time speed tracking in pixels/second
- **Posture**: Game-specific form validation
- **Balance**: Stability and alignment metrics

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new sports
- Improve metric calculations
- Enhance UI/UX
- Fix bugs

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Rahul Singh Bora

## 🙏 Acknowledgments

- MediaPipe by Google for pose detection
- OpenCV community for computer vision tools

