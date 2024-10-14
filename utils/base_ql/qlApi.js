// qlApi.js
const axios = require("axios");

class QLApi {
  constructor(address, id, secret) {
    this.address = address;
    this.id = id;
    this.secret = secret;
    this.valid = true;
    this.login();
  }

  async login() {
    const url = `${this.address}/open/auth/token?client_id=${this.id}&client_secret=${this.secret}`;
    try {
      const response = await axios.get(url);
      if (response.data.code == 200) {
        this.auth = `${response.data.data.token_type} ${response.data.data.token}`;
      } else {
        console.log(`登录失败：${response.data.message}`);
      }
    } catch (e) {
      this.valid = false;
      console.log(`登录失败：${e}`);
    }
  }

  async getEnvs() {
    const url = `${this.address}/open/envs?searchValue=`;
    try {
      const response = await axios.get(url, {
        headers: {
          Authorization: this.auth,
          "Content-Type": "application/json",
        },
      });
      if (response.data.code === 200) {
        return response.data.data;
      } else {
        console.log(`获取环境变量失败：${response.data.message}`);
      }
    } catch (e) {
      console.log(`获取环境变量失败：${e}`);
    }
  }

  async deleteEnvs(ids) {
    const url = `${this.address}/open/envs`;
    try {
      const response = await axios.delete(url, {
        headers: {
          Authorization: this.auth,
          "Content-Type": "application/json",
        },
        data: ids,
      });
      const rjson = response.data;
      if (rjson.code === 200) {
        console.log(`删除环境变量成功：${ids.length}`);
        return true;
      } else {
        console.log(`删除环境变量失败：${rjson.message}`);
        return false;
      }
    } catch (e) {
      console.log(`删除环境变量失败：${e}`);
      return false;
    }
  }

  async addEnvs(envs) {
    const url = `${this.address}/open/envs`;
    try {
      const response = await axios.post(url, envs, {
        headers: {
          Authorization: this.auth,
          "Content-Type": "application/json",
        },
      });
      const rjson = response.data;
      if (rjson.code === 200) {
        console.log(`新建环境变量成功：${envs.length}`);
        return true;
      } else {
        console.log(`新建环境变量失败：${rjson.message}`);
        return false;
      }
    } catch (e) {
      console.log(`新建环境变量失败：${e}`);
      return false;
    }
  }

  async updateEnv(env) {
    const url = `${this.address}/open/envs`;
    try {
      const response = await axios.put(url, env, {
        headers: {
          Authorization: this.auth,
          "Content-Type": "application/json",
        },
      });
      const rjson = response.data;
      if (rjson.code === 200) {
        console.log("更新环境变量成功");
        return true;
      } else {
        console.log(`更新环境变量失败：${rjson.message}`);
        return false;
      }
    } catch (e) {
      console.log(`更新环境变量失败：${e}`);
      return false;
    }
  }
}

module.exports = QLApi;
