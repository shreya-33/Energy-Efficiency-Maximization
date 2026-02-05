# Energy Efficiency Maximization in Smart Homes

## Overview
The "Energy Efficiency Maximization in Smart Homes" project leverages Multi-Agent Q-Learning to optimize energy management across residential spaces. The system intelligently manages the energy consumption of household appliances in response to fluctuations in energy market prices, aiming to reduce costs and enhance system transparency. By increasing user engagement, this project makes smart home energy systems more understandable and trustworthy.

## Repository Structure
### `/backend`
- **`server.py`**: Contains the server-side logic that interacts with the front end, processes data, and executes the Q-learning algorithms.

### `/frontend`
- **`src/`**: React components used to build the user interface.
- **`App.js`**: Main React application file where the components are integrated.
- **`index.js`**: Entry point for the React application.

## Installation
### Prerequisites
- Python 3.8 or later
- Node.js 12.x or later
- npm (Node Package Manager)

### Setup Instructions
1. **Clone the Repository**
   ```
   git clone https://github.com/yourusername/Energy-Efficiency-Maximization-in-Smart-Homes.git
   cd Energy-Efficiency-Maximization-in-Smart-Homes
   ```

2. **Set Up the Backend**
   Navigate to the backend directory and install the required Python packages.
   ```
   cd backend
   ```

3. **Set Up the Frontend**
   Navigate to the frontend directory and install the necessary npm packages.
   ```
   cd ../frontend
   npm install
   ```

## Running the Application
1. **Start the Backend Server**
   ```
   python server.py
   ```
   This will start the backend server on localhost.

2. **Launch the Frontend**
   ```
   npm start
   ```
   This will open the web application in your default browser, typically at `http://localhost:3000`.

## Usage
Once the application is running, navigate through the user interface to configure device settings, view real-time energy consumption data, and receive personalized recommendations for energy efficiency.
