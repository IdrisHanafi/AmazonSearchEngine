import axios from "axios";
import config from "../config.json";

function getProducts({ queryString, categoryId, filterType }) {
  const url = `${config.END_POINT}get_products/?user_query=${queryString}&category_id=${categoryId}&filter_type=${filterType}`;

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

export default getProducts;
