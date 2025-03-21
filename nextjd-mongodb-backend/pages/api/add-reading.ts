import { NextApiRequest, NextApiResponse } from 'next';
import { getServerSession } from 'next-auth';
import authOptions from './auth/[...nextauth]';
import clientPromise from '../../lib/mongodb';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const session = await getServerSession(req, res, authOptions);

  if (!session) {
    console.error('Unauthorized request - No session');
    return res.status(401).json({ message: 'Unauthorized' });
  }

  if (req.method !== 'POST') {
    res.setHeader('Allow', ['POST']);
    return res.status(405).json({ message: `Method ${req.method} Not Allowed` });
  }

  const { location, value } = req.body;

  try {
    const client = await clientPromise;
    const db = client.db('userdata123');
    const collection = db.collection('readings');

    const newReading = { location, value, userId: session.user.id, createdAt: new Date() };
    await collection.insertOne(newReading);

    res.status(201).json({ message: 'Reading added successfully' });
  } catch (error) {
    console.error('Error adding reading:', error);
    res.status(500).json({ message: 'Internal Server Error' });
  }
}
