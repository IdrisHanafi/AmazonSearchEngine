import React, { useState, useEffect } from "react";

// import logo from '../../logo.svg';
import mcrib from '../../mcrib.svg';
import './App.css';

// import services
import getSubcategory from "../../services/getSubcategory";
import getProducts from "../../services/getProducts";

// import components
import SearchAndButton from "../../components/molecules/SearchAndButton/SearchAndButton";
import Filters from "../../components/molecules/Filters/Filters";
import CheckBox from "../../components/atoms/CheckBox/CheckBox";
import LineBreak from "../../components/atoms/LineBreak/LineBreak";
import ProductInfoList from "../../components/organisms/ProductInfoList/ProductInfoList";

function App() {
  const [currText, setText] = useState("");
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [availableCategories, setAvailableCategories] = useState(null);
  const [selectedFilter, setSelectedFilter] = useState(null);
  const [foundProducts, setFoundProducts] = useState(null);

  function resetCurrSelections() {
    setSelectedCategory(null);
    setSelectedFilter(null);
    setAvailableCategories(null)
    setFoundProducts(null)
  }

	function onSubmit() {
    console.log("SUBMITTING");
    console.log(currText);
    getSubcategoryCall(currText);
  }

	function setTextCall(text) {
    resetCurrSelections();
    setText(text);
  }

  function getSubcategoryCall(queryString) {
    getSubcategory({ queryString }).then((res) => {
      console.log("Successfully got response");
      console.log(res);
      const subcategoriesFound = res.subcategory_found;
      setAvailableCategories(subcategoriesFound);
    }).catch((error) => {
      console.log("error");
      console.log(error)
    });
  }

  useEffect(() => {
    // TODO: Only call search when the user press enter
    if (currText && selectedCategory !== null && selectedFilter) {
      const categoryId = availableCategories[selectedCategory].index;
      console.log(currText, categoryId, selectedCategory, selectedFilter);
      getProductsCall(currText, categoryId, selectedFilter);
    }
  }, [currText, selectedCategory, availableCategories, selectedFilter])


  function getProductsCall(queryString, categoryId, filterType) {
    console.log("GETTING PRODUCT API")
    getProducts({ queryString, categoryId, filterType }).then((res) => {
      console.log("Successfully got response");
      console.log(res);
      const productsFound = res.result;
      setFoundProducts(productsFound);
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
          <SearchAndButton currText={currText} setText={setTextCall} onSubmit={onSubmit} />
        </div>

        {availableCategories && (
          <>
            <LineBreak />
            <p style={{ fontWeight: "bold" }}>
            Please confirm which subcategory you're looking for?
            </p>
            <CheckBox 
              selectedIndex={selectedCategory} 
              onClick={onClickCheckBox} 
              items={availableCategories}
            />

            <LineBreak />
            <p style={{ fontWeight: "bold" }}>
            Select your preferred filter:
            </p>
            <Filters 
              selectedFilter={selectedFilter} 
              onClick={onClickFilter}
            />
          </>
        )}

        <ProductInfoList products={foundProducts} />
      </div>

    </div>
  );
}

export default App;
