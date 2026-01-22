"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import Webcam from "react-webcam";
import { supabase } from "@/lib/supabase";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { Camera, Upload, RefreshCcw, CheckCircle2, AlertCircle, BookOpen, Send, Sparkles, Brain } from "lucide-react";
import { RecommendationList } from "@/components/RecommendationList";
import { useRouter } from "next/navigation";
import { Textarea } from "@/components/ui/textarea";
import { AboutEnso } from "@/components/AboutEnso";

const DEFAULT_BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function CapturePage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [image, setImage] = useState<string | null>(null);
  const [user, setUser] = useState<any>(null);
  const webcamRef = useRef<Webcam>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [loadingAuth, setLoadingAuth] = useState(true);
  const [note, setNote] = useState("");
  const [savingNote, setSavingNote] = useState(false);
  const [isSecure, setIsSecure] = useState(true);
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const router = useRouter();

  useEffect(() => {
    // Check if secure context for webcam
    if (typeof window !== "undefined") {
      const host = window.location.hostname;
      const port = window.location.port;
      
      // If we're accessing via IP, update backend URL to use the same IP
      if (host !== "localhost" && host !== "127.0.0.1" && DEFAULT_BACKEND_URL.includes("localhost")) {
        setBackendUrl(`http://${host}:8000`);
      }

      // Allow camera access on localhost, 127.0.0.1, or secure contexts
      const secure = window.isSecureContext || 
                     host === "localhost" || 
                     host === "127.0.0.1" ||
                     (host === "::1" && port === "3001");

      setIsSecure(secure);
      if (!secure) {
        toast.error("Camera access requires a secure connection (HTTPS or localhost).", {
          description: `Please access the app via http://localhost:3001 for camera access. Current URL: ${window.location.href}`,
          duration: 10000,
        });
      } else {
        // Show success message if on camera-friendly port
        if (port === "3001") {
          console.log("Camera access enabled on port 3001");
        }
      }
    }

    supabase.auth.getUser().then(({ data: { user } }) => {
      if (!user) {
        router.push("/auth");
      } else {
        setUser(user);
        setLoadingAuth(false);
      }
    });
  }, [router]);

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      setImage(imageSrc);
      processImage(imageSrc);
    }
  }, [webcamRef]);

  const onUserMediaError = useCallback((error: string | DOMException) => {
    console.error("Webcam error:", error);
    let message = "Failed to access webcam. Please ensure you have granted camera permissions.";
    
    if (error instanceof DOMException) {
      if (error.name === "NotAllowedError") {
        message = "Camera access denied. Please enable camera permissions in your browser settings.";
      } else if (error.name === "NotFoundError") {
        message = "No camera found. Please connect a webcam.";
      } else if (error.name === "NotReadableError") {
        message = "Camera is already in use by another application.";
      } else if (error.name === "OverconstrainedError") {
        message = "Your camera doesn't support the requested resolution.";
      }
    }
    
    toast.error(message);
  }, []);

  const videoConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: "user"
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result as string;
        setImage(base64String);
        processImage(base64String);
      };
      reader.readAsDataURL(file);
    }
  };

  const updateChallenges = async (userId: string, emotion: string) => {
    try {
      // 1. Fetch current challenges
      const { data: challenges, error } = await supabase
        .from("wellness_challenges")
        .select("*")
        .eq("user_id", userId)
        .eq("status", "in_progress");
      
      if (error || !challenges) return;

      for (const challenge of challenges) {
        let newCount = challenge.current_count;
        let shouldUpdate = false;

        if (challenge.category === "mindfulness") {
          // Every capture counts
          newCount += 1;
          shouldUpdate = true;
        } else if (challenge.category === "mood" && emotion === "happy") {
          // Positivity Quest
          newCount += 1;
          shouldUpdate = true;
        } else if (challenge.category === "consistency") {
          // This one is harder to track perfectly without streaks logic, 
          // but let's just increment if it's a new day's capture.
          // For now, simpler: increment every capture but cap at target? 
          // No, let's just increment for now.
          newCount += 1;
          shouldUpdate = true;
        }

        if (shouldUpdate) {
          const isCompleted = newCount >= challenge.target_count;
          await supabase
            .from("wellness_challenges")
            .update({ 
              current_count: newCount,
              status: isCompleted ? "completed" : "in_progress"
            })
            .eq("id", challenge.id);
          
          if (isCompleted) {
            toast.success(`Challenge Completed: ${challenge.title}!`, {
              icon: "🏆"
            });
          }
        }
      }
    } catch (err) {
      console.error("Challenge Update Error:", err);
    }
  };

  const processImage = async (imageSrc: string) => {
    setLoading(true);
    try {
      // Convert base64 to blob
      const res = await fetch(imageSrc);
      const blob = await res.blob();
      
      const formData = new FormData();
      formData.append("file", blob, "capture.jpg");

      // Call FastAPI Backend
      const response = await fetch(`${backendUrl}/detect`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Backend error:", errorText);
        throw new Error(`Failed to detect emotion. Backend returned ${response.status}: ${response.statusText}. Make sure the backend is running on ${backendUrl}`);
      }

      const data = await response.json();
      setResult(data);

      // Save to Supabase with fresh user check
      const { data: { user: currentUser } } = await supabase.auth.getUser();
      
      if (currentUser) {
        console.log("Saving mood entry for user:", currentUser.id);
        const { data: entryData, error: entryError } = await supabase
          .from("mood_entries")
          .insert({
            user_id: currentUser.id,
            emotion: data.emotion,
            mood: data.mood,
            confidence: data.confidence,
            context_data: {
              time_of_day: new Date().getHours(),
              platform: "web"
            }
          })
          .select()
          .single();

        if (entryError) {
          console.error("Supabase Entry Error:", entryError);
          throw entryError;
        }

        // Update Challenges
        await updateChallenges(currentUser.id, data.emotion);

        // Save recommendations
        const recommendationsToSave = data.recommendations.map((rec: any) => ({
          entry_id: entryData.id,
          category: rec.type,
          content: rec.content,
          reason: rec.reason,
          link: rec.link
        }));

        const { error: recError } = await supabase
          .from("recommendations")
          .insert(recommendationsToSave);

        if (recError) {
          console.error("Supabase Rec Error:", recError);
          throw recError;
        }
        
        // Update local state to include the saved entry ID for feedback
        setResult((prev: any) => ({ ...prev, entryId: entryData.id }));
      } else {
        console.warn("No user found, mood entry not saved.");
      }

      toast.success(`Detected ${data.emotion} emotion!`);
    } catch (error: any) {
      console.error(error);
      toast.error(error.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImage(null);
    setResult(null);
    setNote("");
  };

  const [sentiment, setSentiment] = useState<any>(null);

  const saveNote = async () => {
    if (!result || !result.entryId || !note.trim()) return;
    
    setSavingNote(true);
    try {
        // 1. Analyze sentiment via backend
        const response = await fetch(`${backendUrl}/analyze-text`, {
          method: "POST",
          body: JSON.stringify({ text: note.trim() }),
          headers: { "Content-Type": "application/json" }
        });


      let sentimentData = { score: 0, label: "neutral", insight: "" };
      if (response.ok) {
        sentimentData = await response.json();
        setSentiment(sentimentData);
      }

      // 2. Update Supabase
      const { error } = await supabase
        .from("mood_entries")
        .update({ 
          note: note.trim(),
          sentiment_score: sentimentData.score,
          sentiment_label: sentimentData.label
        })
        .eq("id", result.entryId);

      if (error) throw error;
      toast.success("Journal entry saved with AI analysis!");
    } catch (error: any) {
      console.error("Save Note Error:", error);
      toast.error("Failed to save journal entry");
    } finally {
      setSavingNote(false);
    }
  };

    const updateEmotion = async (newEmotion: string) => {
    if (!result || !result.entryId) {
      console.warn("No entry ID found for update");
      toast.error("Please capture or upload an image first");
      return;
    }
    
    setLoading(true);
    try {
      console.log(`Updating emotion to ${newEmotion} for entry ${result.entryId}`);
      
      // Get new recommendations from backend for the manual emotion
      const response = await fetch(`${backendUrl}/manual-update`, {
        method: "POST",
        body: JSON.stringify({ emotion: newEmotion }),
        headers: { "Content-Type": "application/json" }
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Unknown backend error" }));
        throw new Error(errorData.detail || "Failed to get updated recommendations");
      }
      
      const data = await response.json();

      // Update mood entry in Supabase
      const { error: entryError } = await supabase
        .from("mood_entries")
        .update({ 
          emotion: newEmotion,
          mood: data.mood,
          confidence: 1.0 
        })
        .eq("id", result.entryId);

      if (entryError) {
        console.error("Supabase Update Error:", entryError);
        throw new Error(`Database update failed: ${entryError.message}`);
      }

      // Delete old recommendations and save new ones
      const { error: deleteError } = await supabase
        .from("recommendations")
        .delete()
        .eq("entry_id", result.entryId);
      
      if (deleteError) {
        console.error("Supabase Delete Error:", deleteError);
        // We continue even if delete fails, though it shouldn't
      }

      const recommendationsToSave = data.recommendations.map((rec: any) => ({
        entry_id: result.entryId,
        category: rec.type,
        content: rec.content,
        reason: rec.reason,
        link: rec.link
      }));

      const { error: recError } = await supabase
        .from("recommendations")
        .insert(recommendationsToSave);

      if (recError) {
        console.error("Supabase Insert Error:", recError);
        throw new Error(`Failed to save new recommendations: ${recError.message}`);
      }

      setResult((prev: any) => ({
        ...prev,
        emotion: newEmotion,
        mood: data.mood,
        confidence: 1.0,
        recommendations: data.recommendations
      }));

      // Update Challenges for manual update
      const { data: { user: currentUser } } = await supabase.auth.getUser();
      if (currentUser) {
        await updateChallenges(currentUser.id, newEmotion);
      }
      
      toast.success(`Updated to ${newEmotion} and refreshed recommendations!`);
    } catch (error: any) {
      console.error("Update Emotion Error:", error);
      toast.error(error.message || "Failed to update emotion");
    } finally {
      setLoading(false);
    }
  };

  if (loadingAuth) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <RefreshCcw className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center space-y-2 relative">
        <div className="absolute right-0 top-0">
          <AboutEnso />
        </div>
        <h1 className="text-4xl font-bold tracking-tight">How are you feeling today?</h1>
        <p className="text-muted-foreground">Capture your expression or upload a photo for AI-driven wellness insights.</p>
      </div>

      {!result ? (
        <Card className="overflow-hidden border-2 border-dashed border-muted-foreground/20">
          <CardContent className="p-0">
            <Tabs defaultValue="webcam" className="w-full">
              <TabsList className="w-full rounded-none h-12">
                <TabsTrigger value="webcam" className="flex-1 gap-2">
                  <Camera className="w-4 h-4" />
                  Webcam
                </TabsTrigger>
                <TabsTrigger value="upload" className="flex-1 gap-2">
                  <Upload className="w-4 h-4" />
                  Upload Image
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="webcam" className="m-0 p-8 flex flex-col items-center gap-6">
                <div className="relative rounded-2xl overflow-hidden bg-black aspect-video w-full max-w-xl border shadow-xl">
                  {isSecure ? (
                    <Webcam
                      audio={false}
                      ref={webcamRef}
                      screenshotFormat="image/jpeg"
                      className="w-full h-full object-cover"
                      videoConstraints={videoConstraints}
                      onUserMediaError={onUserMediaError}
                    />
                  ) : (
                    <div className="absolute inset-0 bg-muted flex flex-col items-center justify-center p-6 text-center gap-4">
                      <AlertCircle className="w-12 h-12 text-destructive" />
                      <div className="space-y-2">
                        <p className="font-bold text-lg">Camera Access Not Available</p>
                        <p className="text-sm text-muted-foreground">
                          Camera access requires a secure connection (HTTPS or localhost). 
                          Please access the app via the camera-friendly URL.
                        </p>
                        <div className="flex flex-col gap-2 mt-4">
                          <Button 
                            variant="default" 
                            size="sm" 
                            onClick={() => window.location.href = "http://localhost:3001"}
                            className="w-full"
                          >
                            Open Camera-Friendly URL (Port 3001)
                          </Button>
                          <p className="text-xs text-muted-foreground">
                            Or use: <code className="bg-muted px-1 py-0.5 rounded">bun run dev:camera</code>
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                  {loading && (
                    <div className="absolute inset-0 bg-background/50 backdrop-blur-sm flex items-center justify-center">
                      <div className="flex flex-col items-center gap-4">
                        <RefreshCcw className="w-8 h-8 animate-spin text-primary" />
                        <p className="font-medium">Analyzing emotion...</p>
                      </div>
                    </div>
                  )}
                </div>
                <Button size="lg" onClick={capture} disabled={loading} className="px-8 gap-2">
                  <Camera className="w-5 h-5" />
                  Capture & Analyze
                </Button>
              </TabsContent>

              <TabsContent value="upload" className="m-0 p-8 flex flex-col items-center gap-6">
                <div 
                  className="w-full max-w-xl aspect-video border-2 border-dashed rounded-2xl flex flex-col items-center justify-center gap-4 cursor-pointer hover:bg-muted/50 transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    className="hidden" 
                    accept="image/*" 
                    onChange={handleFileUpload} 
                  />
                  <div className="p-4 rounded-full bg-primary/10">
                    <Upload className="w-8 h-8 text-primary" />
                  </div>
                  <div className="text-center">
                    <p className="font-medium text-lg">Click to upload or drag and drop</p>
                    <p className="text-sm text-muted-foreground">PNG, JPG or JPEG (max. 5MB)</p>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      ) : (
        <div className="grid md:grid-cols-3 gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="md:col-span-1 space-y-6">
            <Card className="overflow-hidden">
              <div className="aspect-square relative">
                {image && <img src={image} alt="Capture" className="w-full h-full object-cover" />}
                <div className="absolute bottom-4 right-4 bg-background/80 backdrop-blur-md px-3 py-1 rounded-full text-xs font-bold border flex items-center gap-1.5">
                  <CheckCircle2 className="w-3 h-3 text-green-500" />
                  Analyzed
                </div>
              </div>
              <CardHeader className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Detected Emotion</span>
                  <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-primary/10 text-primary">
                    {Math.round(result.confidence * 100)}% Confidence
                  </span>
                </div>
                <CardTitle className="text-3xl capitalize text-primary">{result.emotion}</CardTitle>
                <CardDescription className="text-base font-medium">
                  Resulting Mood: <span className="text-foreground">{result.mood}</span>
                </CardDescription>
              </CardHeader>
              <CardContent className="p-4 pt-0 space-y-4">
                <div className="space-y-2">
                  <p className="text-[10px] font-bold uppercase text-muted-foreground">Not correct? Adjust manually:</p>
                  <div className="flex flex-wrap gap-1.5">
                    {["happy", "sad", "fear", "disgust", "angry", "surprise", "neutral"].map((emo) => (
                      <Button
                        key={emo}
                        variant={result.emotion === emo ? "default" : "outline"}
                        size="sm"
                        className="h-7 px-2 text-[10px] capitalize"
                        onClick={() => updateEmotion(emo)}
                        disabled={loading}
                      >
                        {emo}
                      </Button>
                    ))}
                  </div>
                </div>

                <Button variant="outline" className="w-full gap-2" onClick={reset}>
                  <RefreshCcw className="w-4 h-4" />
                  New Analysis
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="p-4 pb-2">
                <div className="flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-primary" />
                  <CardTitle className="text-sm">Mood Journal</CardTitle>
                </div>
                <CardDescription className="text-[11px]">
                  What's causing this {result.emotion} feeling?
                </CardDescription>
              </CardHeader>
              <CardContent className="p-4 pt-0 space-y-3">
                <Textarea 
                  placeholder="e.g. Just finished a big project, rainy day, feeling sleepy..."
                  className="text-xs min-h-[80px] resize-none"
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                />
                {sentiment && (
                  <div className="p-2 rounded bg-primary/5 border border-primary/10 space-y-1 animate-in fade-in slide-in-from-top-1">
                    <div className="flex items-center justify-between">
                      <span className="text-[9px] font-bold uppercase text-primary">AI Sentiment: {sentiment.label}</span>
                      <Sparkles className="w-3 h-3 text-primary" />
                    </div>
                    <p className="text-[10px] text-muted-foreground leading-tight italic">
                      "{sentiment.insight}"
                    </p>
                  </div>
                )}
                <Button 
                  className="w-full h-8 text-xs gap-2" 
                  size="sm"
                  onClick={saveNote}
                  disabled={savingNote || !note.trim()}
                >
                  {savingNote ? <RefreshCcw className="w-3 h-3 animate-spin" /> : <Send className="w-3 h-3" />}
                  {sentiment ? "Update Entry" : "Save Note"}
                </Button>
              </CardContent>
            </Card>

            <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 flex flex-col gap-3">
              <div className="flex gap-3">
                <AlertCircle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                <p className="text-xs text-amber-500/80 leading-relaxed">
                  {result.disclaimer}
                </p>
              </div>
              {(result.emotion === "angry" || result.emotion === "sad" || result.emotion === "fear") && (
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full text-xs border-amber-500/30 hover:bg-amber-500/10 text-amber-600 dark:text-amber-400 gap-2"
                  onClick={() => router.push("/mindfulness")}
                >
                  <Brain className="w-3 h-3" />
                  Try a breathing exercise
                </Button>
              )}
            </div>
          </div>

          <div className="md:col-span-2">
            <RecommendationList recommendations={result.recommendations} />
          </div>
        </div>
      )}
    </div>
  );
}
