import axios from "axios";
import config from "../config.json";

const modelEndpoints = {
  "M2": "get_subcategory/",
  "M2+": "get_subcategory/v2/",
  "M1": "get_subcategory/m1/",
}

function getSubcategory({ selectedSubcategoryModel, queryString }) {
  const url = `${config.END_POINT}${modelEndpoints[selectedSubcategoryModel]}${queryString}`;

  return new Promise((resolve, reject) => {
    axios.get(url).then((res) => {
      console.log("Get response");
      console.log(res);
      resolve(res.data);
    }).catch((error) => {
      console.log("Get error");
      console.log(error);
      reject(error);
    });
  });
}

export default getSubcategory;
