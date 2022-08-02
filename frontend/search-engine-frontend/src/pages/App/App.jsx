import React, { useState } from "react";

// import logo from '../../logo.svg';
import mcrib from '../../mcrib.svg';
import './App.css';

// import services
import getSubcategory from "../../services/getSubcategory";

// import atoms
import SearchAndButton from "../../components/molecules/SearchAndButton/SearchAndButton";
import Filters from "../../components/molecules/Filters/Filters";
import CheckBox from "../../components/atoms/CheckBox/CheckBox";

const items = [
  { 
    label: "Subcategory 1",
    index: 200,
  },
  { 
    label: "Subcategory 2",
    index: 202,
  },
  { 
    label: "Subcategory 3",
    index: 203,
  },
];

function App() {
  const [currText, setText] = useState("");
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [selectedFilter, setSelectedFilter] = useState(null);

	function onSubmit() {
    console.log("HELLO");
    console.log(currText);
    getSubcategoryCall(currText);
  }

  function getSubcategoryCall(queryString) {
    getSubcategory({ queryString }).then((res) => {
      console.log("Successfully got response");
      console.log(res);
    }).catch((error) => {
      console.log("error");
      console.log(error)
    });
  }

  function onClickCheckBox(indexPressed) {
    console.log(`clicked checkbox ${indexPressed}!`);
    if (indexPressed === selectedCategory) {
      setSelectedCategory(null)
    } else {
      setSelectedCategory(indexPressed)
    }
  }

  function onClickFilter(newSelectedFilter) {
    console.log(`clicked filter: ${newSelectedFilter}!`);
    if (selectedFilter === newSelectedFilter) {
      setSelectedFilter(null)
    } else {
      setSelectedFilter(newSelectedFilter)
    }
  }

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
          <SearchAndButton currText={currText} setText={setText} onSubmit={onSubmit} />
        </div>

        <p>
        Please confirm which subcategory you're looking for?
        </p>
        <CheckBox selectedIndex={selectedCategory} onClick={onClickCheckBox} items={items}/>

        <p>
        Select your preferred filter:
        </p>
        <Filters selectedFilter={selectedFilter} onClick={onClickFilter}/>
      </div>

    </div>
  );
}

export default App;
