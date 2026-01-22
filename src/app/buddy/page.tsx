"use client";

import { useState, useEffect, useRef } from "react";
import { supabase } from "@/lib/supabase";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { toast } from "sonner";
import { Send, MessageCircle, Sparkles, User, Bot, RefreshCcw } from "lucide-react";
import { useRouter } from "next/navigation";

const DEFAULT_BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [lastMood, setLastMood] = useState<string | null>(null);
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const scrollRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  useEffect(() => {
    if (typeof window !== "undefined") {
      const host = window.location.hostname;
      if (host !== "localhost" && host !== "127.0.0.1" && DEFAULT_BACKEND_URL.includes("localhost")) {
        setBackendUrl(`http://${host}:8000`);
      }
    }

    supabase.auth.getUser().then(({ data: { user } }) => {
      if (!user) {
        router.push("/auth");
      } else {
        setUser(user);
        fetchMessages(user.id);
        fetchLastMood(user.id);
      }
    });
  }, [router]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const fetchMessages = async (userId: string) => {
    const { data, error } = await supabase
      .from("chat_messages")
      .select("*")
      .eq("user_id", userId)
      .order("created_at", { ascending: true });

    if (error) {
      console.error("Fetch Messages Error:", error);
    } else {
      setMessages(data || []);
    }
  };

  const fetchLastMood = async (userId: string) => {
    const { data, error } = await supabase
      .from("mood_entries")
      .select("emotion")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(1)
      .single();

    if (data) setLastMood(data.emotion);
  };

  const sendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || !user || loading) return;

    const userMessage = input.trim();
    setInput("");
    setLoading(true);

    try {
      // 1. Save user message to Supabase
      const { data: savedUserMsg, error: userMsgError } = await supabase
        .from("chat_messages")
        .insert({
          user_id: user.id,
          role: "user",
          content: userMessage
        })
        .select()
        .single();

      if (userMsgError) throw userMsgError;
      setMessages((prev) => [...prev, savedUserMsg]);

      // 2. Call Backend AI
      const response = await fetch(`${backendUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          current_mood: lastMood,
          history: messages.slice(-5).map(m => ({ role: m.role, content: m.content }))
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Backend error:", errorText);
        throw new Error(`Failed to get response from Enso Buddy: ${response.status} ${response.statusText}. Make sure the backend is running on ${backendUrl}`);
      }
      
      const data = await response.json();
      const aiContent = data.response || "I'm here for you. Can you tell me more?";

      // 3. Save assistant message to Supabase
      const { data: savedAiMsg, error: aiMsgError } = await supabase
        .from("chat_messages")
        .insert({
          user_id: user.id,
          role: "assistant",
          content: aiContent
        })
        .select()
        .single();

      if (aiMsgError) throw aiMsgError;
      setMessages((prev) => [...prev, savedAiMsg]);

    } catch (error: any) {
      console.error("Chat Error:", error);
      toast.error(error.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto h-[calc(100vh-12rem)] flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-primary/10">
            <MessageCircle className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Enso Buddy</h1>
            <p className="text-sm text-muted-foreground">Your AI wellness companion</p>
          </div>
        </div>
        {lastMood && (
          <div className="px-3 py-1 rounded-full bg-muted border text-xs font-medium flex items-center gap-2">
            <Sparkles className="w-3 h-3 text-primary" />
            Current Context: <span className="capitalize text-primary">{lastMood}</span>
          </div>
        )}
      </div>

      <Card className="flex-1 flex flex-col overflow-hidden border-2">
        <CardContent className="flex-1 overflow-hidden p-0 relative">
          <ScrollArea className="h-full p-4" ref={scrollRef}>
            <div className="space-y-4">
              {messages.length === 0 && !loading && (
                <div className="flex flex-col items-center justify-center py-12 text-center space-y-4">
                  <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
                    <Bot className="w-6 h-6 text-muted-foreground" />
                  </div>
                  <div className="max-w-xs">
                    <p className="font-medium">Welcome! I'm Enso Buddy.</p>
                    <p className="text-sm text-muted-foreground">I can help you process your emotions, give you mindfulness tips, or just listen. Say hello!</p>
                  </div>
                </div>
              )}
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground rounded-tr-none"
                        : "bg-muted border rounded-tl-none"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1 opacity-70">
                      {msg.role === "user" ? (
                        <User className="w-3 h-3" />
                      ) : (
                        <Bot className="w-3 h-3" />
                      )}
                      <span className="text-[10px] font-bold uppercase">
                        {msg.role === "user" ? "You" : "Enso Buddy"}
                      </span>
                    </div>
                    <p className="leading-relaxed">{msg.content}</p>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-muted border rounded-2xl rounded-tl-none px-4 py-3 text-sm animate-pulse flex items-center gap-2">
                    <RefreshCcw className="w-3 h-3 animate-spin" />
                    <span>Enso is thinking...</span>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
        <CardFooter className="p-4 border-t bg-muted/30">
          <form onSubmit={sendMessage} className="flex w-full gap-2">
            <Input
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              className="bg-background"
            />
            <Button type="submit" disabled={loading || !input.trim()} size="icon">
              <Send className="w-4 h-4" />
            </Button>
          </form>
        </CardFooter>
      </Card>
      <p className="text-[10px] text-center text-muted-foreground italic">
        Enso Buddy uses your mood history to provide context. Conversation is private to your account.
      </p>
    </div>
  );
}
