// simple axios wrapper — adjust baseURL if your backend runs on different host/port
import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 15000,
});

export default API;
