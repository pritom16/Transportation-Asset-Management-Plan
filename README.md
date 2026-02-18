# Transportation Asset Management (TAM) Dashboard

A Flask-based geospatial analysis dashboard for transportation infrastructure using OpenStreetMap data, network analysis, and centrality metrics.

## Features

- **OSMnx Analysis**: Extract and analyze street networks from OpenStreetMap
- **GIS Analysis**: Process geographic data files (shapefiles, GeoJSON, etc.)
- **Centrality Analysis**: Compute network centrality metrics (betweenness, closeness, etc.)
- **Interactive Visualizations**: Web-based maps and statistics using Folium and Plotly
- **Large File Support**: Handle up to 500 MB file uploads

## Technologies

- **Backend**: Flask, Flask-CORS
- **Geospatial**: GeoPandas, OSMnx, NetworkX, Momepy
- **Visualization**: Folium, Matplotlib, Contextily, Plotly
- **Frontend**: HTML5, Bootstrap, JavaScript

## Local Setup

### Prerequisites
- Python 3.11+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pritom16/Transportation-Asset-Management-Plan.git
   cd Transportation-Asset-Management-Plan
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Access the dashboard**
   - Open http://localhost:5000 in your browser
   - Centrality analysis: http://localhost:5000/centrality

## Deployment on Render

### Step 1: Push to GitHub

Make sure all files are committed and pushed:
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### Step 2: Deploy to Render

1. **Create a Render account** at [render.com](https://render.com)

2. **Connect your GitHub repository**
   - Go to Dashboard → New → Web Service
   - Select "Build and deploy from a Git repository"
   - Choose your Transportation-Asset-Management-Plan repository

3. **Configure the service**
   - **Name**: `tam-dashboard` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn main:app`
   - **Instance Type**: Choose appropriate tier (Standard or higher for geospatial processing)

4. **Add Environment Variables** (Optional, if needed)
   - FLASK_ENV: `production`

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete (usually 5-10 minutes)
   - Your app will be available at: `https://tam-dashboard.onrender.com`

### Step 3: Access Your Dashboard

- **Main Dashboard**: https://tam-dashboard.onrender.com
- **Centrality Analysis**: https://tam-dashboard.onrender.com/centrality

## API Endpoints

### 1. OSMnX Analysis
- **POST** `/api/analyze_osmnx`
- **Parameters**:
  - `location`: Location name (e.g., "Atlanta, GA")
  - `network_type`: Type of network (`drive`, `walk`, `bike`, etc.)
- **Returns**: Network visualization and statistics

### 2. GIS Analysis
- **POST** `/api/analyze_gis`
- **Parameters**:
  - File upload with geospatial data
  - `file`: GeoJSON, Shapefile, or GDB format
- **Returns**: Processed GIS visualization and analysis

### 3. Centrality Analysis
- **POST** `/api/analyze_centrality`
- **Parameters**:
  - `location`: Location name
  - `network_type`: Type of network
  - `centrality_measure`: Measure type (`betweenness`, `closeness`, `degree`, etc.)
- **Returns**: Centrality visualization and top elements

## Important Notes

### Large Files
- The `.gitignore` file excludes large geodatabases from version control
- Data files (`.gdb`, `.zip`) are stored locally but not committed
- For production, consider using cloud storage (S3, Dropbox) for large datasets

### Performance Considerations
- Geospatial analysis is computationally intensive
- For large datasets or frequent requests, consider upgrading your Render instance
- Process timeout is set to accommodate long-running analyses

### CORS Configuration
- The app allows CORS requests from any origin for flexibility
- For production, restrict to specific domains if needed

## File Structure

```
.
├── main.py                  # Flask application
├── index.html              # Main dashboard page
├── centrality.html         # Centrality analysis page
├── requirements.txt        # Python dependencies
├── Procfile               # Render deployment configuration
├── runtime.txt            # Python version specification
├── .render-build.sh       # Build script for Render
└── uploads/               # Uploaded file storage
```

## Troubleshooting

### Deployment Issues

**Issue**: "Build failed"
- **Solution**: Check that all dependencies in `requirements.txt` are available
- Ensure Python 3.11+ is specified in `runtime.txt`

**Issue**: "Timeout error during deployment"
- **Solution**: Some geospatial packages need extra time to compile
- Try increasing the Render instance tier

**Issue**: CORS errors in browser console
- **Solution**: The CORS is set to allow all origins. If needed, update main.py line 22

### Runtime Issues

**Issue**: "Module not found" errors
- **Solution**: Run `pip install -r requirements.txt` locally and update the file

**Issue**: Map visualization not loading
- **Solution**: Check browser console for API errors
- Verify Folium and contextily versions match locally

## Development Tips

1. **Local Testing**: Always test locally before deploying
2. **Debug Mode**: Set `FLASK_ENV=development` for debug logging
3. **Large File Processing**: For testing large files, increase timeout in Render dashboard settings
4. **Monitoring**: Check Render logs in dashboard → Logs tab

## Support

For issues or questions:
- Check the GitHub repository: https://github.com/pritom16/Transportation-Asset-Management-Plan
- Review Render documentation: https://render.com/docs
- Consult Flask documentation: https://flask.palletsprojects.com

## License

Add your license information here.
