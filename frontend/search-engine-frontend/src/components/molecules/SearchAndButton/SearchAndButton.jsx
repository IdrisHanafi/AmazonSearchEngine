import React from "react";

import "./SearchAndButton.css";

// import atoms
import Button from "../../atoms/Button/Button";
import Input from "../../atoms/Input/Input";

function SearchAndButton({ currText, setText, onSubmit }) {

  function handleTextChange(e) {
    setText(e.target.value);
  }

  function handleKeypress(e) {
		// it triggers by pressing the enter key
    if (e.keyCode === 13) {
      console.log("enter pressed");
      onSubmit();
    }
  }

  return (
    <div className="search">
      <Input 
        type="text" 
        className="searchTerm" 
        placeholder="What are you looking for?" 
        value={currText} 
        onChange={handleTextChange} 
        onKeyDown={handleKeypress}
      />
      <Button 
        type="submit" 
        className="searchButton" 
        onClick={onSubmit}
      >
        <i className="fa fa-search"></i>
     </Button>
   </div>
  );
}

export default SearchAndButton;
