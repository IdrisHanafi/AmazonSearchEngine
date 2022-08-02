import React from "react";

// import logo from '../../logo.svg';
import mcrib from '../../mcrib.svg';
import './App.css';

// import atoms
import SearchAndButton from "../../components/molecules/SearchAndButton/SearchAndButton";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={mcrib} className="App-logo" alt="logo" /> 
      </header>

      <div className="contentBox">
        <p>
          Amazon Product Search Engine
        </p>
        <div className="parentSearch" >
          <SearchAndButton />
        </div>
      </div>

    </div>
  );
}

export default App;
