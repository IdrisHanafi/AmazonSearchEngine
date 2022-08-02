import axios from "axios";
import config from "../config.json";

function getSubcategory({ queryString }) {
  const url = `${config.END_POINT}get_subcategory/${queryString}`;

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
