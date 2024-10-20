import React, { useState } from 'react';
import './App.css';

function App() {
  const [imgfile, userimgFile] = useState(null);

  // Handle file selection, if changing files
  const handleFileChange = (event) => {
    userimgFile(event.target.files[0]);
  };

  // Handle form submission
  const handleSubmit = (event) => {
    event.preventDefault(); //prevents refreshing every upload
    if (!imgfile) {
      alert('No file selected');
      return;
    }

    // FormData to send the file to the server
    const formData = new FormData();
    formData.append('file', imgfile);

    // Make a POST request to the backend
    fetch('your-backend-endpoint', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log('File uploaded successfully:', data);
      })
      .catch((error) => {
        console.error('Error uploading file:', error);
      });
  };

  return (
    <div id='upload'>
      <h1>File Uploader</h1>
      <form id="uploadForm" onSubmit={handleSubmit}>
        <div id="drop-area">
          Drop files here to upload
          </div>
          <div id="file-input-wrapper">
            <input type="file" name="file" id="fileinput" accept="image/*" onChange={handleFileChange}/>
            <button type="submit">Upload File</button>
          </div>
      </form>
    </div>
  );
}

export default App;
