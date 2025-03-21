"use client";

import React, { Fragment } from 'react';
import { useState } from 'react';
import { useSession, signIn } from 'next-auth/react';
import { useRouter } from 'next/router';
import { useEffect } from 'react';
import { signOut } from 'next-auth/react';

export default function AddReading() {
  const { data: session } = useSession();
  const router = useRouter();
  const [form, setForm] = useState({ location: '', value: '' });

  useEffect(() => {
    console.log('Session from useSession():', session);
  }, [session]);

  if (!session) {
    return (
      <Fragment>
        <div>
          <p>You need to be authenticated to add readings.</p>
          <button onClick={() => signIn()}>Sign in</button>
        </div>
      </Fragment>
    );
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch('/api/add-reading', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.accessToken}`, // Ensure the session token is sent with the request
          credentials: 'include', // This ensures cookies are sent with the request
        },
        body: JSON.stringify({ ...form, userId: session.user.id }),
      });
      if (res.ok) {
        router.push('/');
      } else {
        console.error('Failed to add reading');
      }
    } catch (error) {
      console.error('Error adding reading:', error);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-8">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <input
          type="text"
          placeholder="Location"
          value={form.location}
          onChange={(e) => setForm({ ...form, location: e.target.value })}
          required
        />
        <input
          type="text"
          placeholder="Value"
          value={form.value}
          onChange={(e) => setForm({ ...form, value: e.target.value })}
          required
        />
        <button type="submit">Add Reading</button>

        <button onClick={() => signOut()}>Logout</button>;
      </form>
    </div>
  );
}
