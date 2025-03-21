import { NextApiRequest, NextApiResponse } from 'next';
import clientPromise from '../../lib/mongodb';
import { ObjectId } from 'mongodb';
import { hash } from 'bcryptjs';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    const client = await clientPromise;
    const db = client.db('userdata123');
    const collection = db.collection('nodejsserver');

    switch (req.method) {
      case 'GET':
        const users = await collection.find({}).toArray();
        res.status(200).json(users);
        break;
      case 'POST':
        const { name, email, password } = req.body;
        const hashedPassword = await hash(password, 10);
        const newUser = { name, email, password: hashedPassword };
        await collection.insertOne(newUser);
        res.status(201).json(newUser);
        break;
      case 'PUT':
        const { id, ...updateData } = req.body;
        await collection.updateOne({ _id: new ObjectId(id) }, { $set: updateData });
        res.status(200).json(updateData);
        break;
      case 'DELETE':
        const { id: deleteId } = req.body;
        await collection.deleteOne({ _id: new ObjectId(deleteId) });
        res.status(204).end();
        break;
      default:
        res.setHeader('Allow', ['GET', 'POST', 'PUT', 'DELETE']);
        res.status(405).end(`Method ${req.method} Not Allowed`);
    }
  } catch (error) {
    console.error('Error connecting to MongoDB:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
}