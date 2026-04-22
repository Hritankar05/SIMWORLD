import express from 'express';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());

const NVIDIA_URL = 'https://integrate.api.nvidia.com/v1/chat/completions';
const API_KEY = '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';

app.post('/api/chat', async (req, res) => {
  try {
    const response = await fetch(NVIDIA_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`,
      },
      body: JSON.stringify(req.body),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error('NVIDIA API error:', response.status, errText);
      return res.status(response.status).json({ error: errText });
    }

    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

app.listen(3001, () => {
  console.log('✅ NVIDIA proxy server running on http://localhost:3001');
});
