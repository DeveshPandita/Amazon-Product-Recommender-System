import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [input, setInput] = useState("");
  const [recommendations, setRecommendations] = useState([]);

  const handleSearch = async () => {
    try {
      const response = await axios.post("http://localhost:5000/api/recommend", {
        input,
      });
      setRecommendations(response.data);
    } catch (error) {
      console.error("Error fetching recommendations:", error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Product Recommender</h1>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter search term"
        />
        <button onClick={handleSearch}>Search</button>
        <div>
          {recommendations.map((item, index) => (
            <div key={index}>
              <h2>
                <a href={`https://www.amazon.com/dp/${item.asin}`}>
                  {item.title}{" "}
                </a>
              </h2>
              <img src={item.imageURLHighRes} alt="No image available" />
            </div>
          ))}
        </div>
      </header>
    </div>
  );
}

export default App;
