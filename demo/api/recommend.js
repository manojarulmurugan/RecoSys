const { GoogleAuth } = require('google-auth-library');

const SERVICE_URL = 'https://recosys-recommender-o34zzoh3da-uc.a.run.app';

// Cache the auth client across warm invocations of the same function instance.
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
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const client = await getClient();
    const upstream = await client.request({
      url: `${SERVICE_URL}/recommend`,
      method: 'POST',
      data: req.body,
    });
    res.status(200).json(upstream.data);
  } catch (err) {
    const status = err.response?.status ?? 500;
    res.status(status).json(err.response?.data ?? { error: err.message });
  }
};
