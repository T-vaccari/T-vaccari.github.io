---
title: PAC Investment Analysis webpage
description: This project aims to build a page where you can see how effective can be to invest with the PAC approach.
link : https://pac-analysis.tommasovaccari.com
layout: project
---
## Project Overview

This project is a web application designed to illustrate the benefits of investing using the PAC (Capital Accumulation Plan) approach. It provides users with a flexible tool to set various investment parameters and visualize the potential outcomes over time. You can reach the page here : [pac-analysis.tommasovaccari.com](https://pac-analysis.tommasovaccari.com)

The frontend of the application is built using traditional web technologies, offering an intuitive and user-friendly interface where users can input key investment parameters such as:

- Choosing the index
- Initial capital
- Monthly contribution : low stake and high stake
- Duration of the investment
- Inflaction

Once the parameters are set, the frontend interacts with a backend API developed in Flask. The backend processes these inputs and executes the core investment logic, which involves calculating the growth of the investment using the PAC approach. The backend performs the necessary computations and returns detailed results, including:

- The total invested amount
- Returns over time
- Probability of losing money
- Average Capital Gain
- Graphical representations of the investment’s growth

The output from the backend is displayed on the frontend as both numerical results and dynamic charts, allowing users to easily understand the long-term benefits of their investment strategy. To perform the calculus of the growth the API uses index historical data using Yahoo Finance API.

## Technical Details

- **Frontend**: The web interface is constructed using modern HTML, CSS, and JavaScript technologies, ensuring compatibility across different devices and screen sizes. The frontend is responsible for collecting user inputs and displaying the investment simulation results.
  
- **Backend (Flask API)**: The backend logic is built using Flask, a lightweight Python framework. The API is responsible for:
  1. Accepting the investment parameters from the frontend.
  2. Running the PAC investment calculation logic.
  3. Returning a JSON response with the computed data, including the graph data points for visualization.
  
- **Data Visualization**: The results are shown in graphical form using a charting library (pyplot). This allows the user to visualize the growth of their investments over time, making it easier to comprehend the benefits of long-term investing via PAC.

The project aims to offer a simple yet powerful platform to help users understand how periodic investments can grow over time, providing them with the insights needed to make informed financial decisions.
