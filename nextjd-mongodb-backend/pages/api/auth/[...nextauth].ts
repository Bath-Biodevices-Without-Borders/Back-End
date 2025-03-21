import NextAuth from 'next-auth';
import CredentialsProvider from 'next-auth/providers/credentials';
import clientPromise from '../../../lib/mongodb';
import { compare } from 'bcryptjs';

export default NextAuth({
  providers: [
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        try {
          const client = await clientPromise;
          const db = client.db('userdata123');
          const collection = db.collection('nodejsserver');

          console.log('Authorizing user:', credentials.email);

          const user = await collection.findOne({ email: credentials.email });
          if (!user) {
            console.error('User not found:', credentials.email);
            throw new Error('Invalid email or password');
          }

          console.log('User found:', user);

          const isValidPassword = await compare(credentials.password, user.password);
          if (!isValidPassword) {
            console.error('Invalid password for user:', credentials.email);
            throw new Error('Invalid email or password');
          }

          console.log('User authorized:', credentials.email);
          return { id: user._id, email: user.email };
        } catch (error) {
          console.error('Error in authorize function:', error);
          throw new Error('Internal Server Error');
        }
      },
    }),
  ],
  session: {
    jwt: true,
  },
  callbacks: {
    async session({ session, token }) {
      session.user.id = token.sub;
      session.accessToken = token.accessToken; // Ensure the access token is included in the session
      return session;
    },
    async jwt({ token, user }) {
      if (user) {
        token.sub = user.id;
        token.accessToken = user.accessToken; // Ensure the access token is included in the token
      }
      return token;
    },
  },
});