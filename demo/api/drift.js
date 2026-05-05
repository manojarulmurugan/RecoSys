const { GoogleAuth } = require('google-auth-library');

const SERVICE_URL = 'https://recosys-recommender-o34zzoh3da-uc.a.run.app';

let _client = null;

async function getClient() {
  if (_client) return _client;
  const credentials = JSON.parse(process.env.GCP_SA_KEY.replace(/^﻿/, ''));
  const auth = new GoogleAuth({ credentials });
  _client = await auth.getIdTokenClient(SERVICE_URL);
  return _client;
}

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');

  try {
    const client = await getClient();
    const upstream = await client.request({ url: `${SERVICE_URL}/drift` });
    res.status(200).json(upstream.data);
  } catch (err) {
    const status = err.response?.status ?? 503;
    res.status(status).json(err.response?.data ?? { error: err.message });
  }
};
