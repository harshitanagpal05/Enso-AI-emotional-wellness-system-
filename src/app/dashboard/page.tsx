"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  BarChart, Bar, PieChart, Pie, Cell, Legend
} from "recharts";
import { 
  Brain, TrendingUp, Heart, MessageSquare, Loader2, Sparkles, 
  Calendar, BookOpen, Clock, AlertCircle, ExternalLink, Music, Film, Activity,
  Search, Filter, Flame, Sun, Sunset, Moon, Sunrise, Trophy, Target, CheckCircle
} from "lucide-react";
import { useRouter } from "next/navigation";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { 
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue 
} from "@/components/ui/select";

const COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#0088fe", "#00C49F", "#FFBB28"];

const WELLNESS_TIPS: Record<string, string[]> = {
  "happy": [
    "Channel this energy into a creative project today!",
    "Share your positivity - compliment three people you encounter.",
    "Take a moment to write down what exactly made you feel this way."
  ],
  "sad": [
    "It's okay to feel this way. Try a 5-minute guided meditation.",
    "A short walk in nature can help reset your perspective.",
    "Don't hesitate to reach out to a friend for a quick chat."
  ],
  "neutral": [
    "A perfect time for focused work or learning something new.",
    "Try a new healthy recipe for dinner tonight.",
    "Consider starting a new book or podcast."
  ],
  "angry": [
    "High-intensity exercise is a great way to release this tension.",
    "Try the 4-7-8 breathing technique: inhale for 4, hold for 7, exhale for 8.",
    "Write down what's bothering you, then physically tear up the paper."
  ],
  "surprise": [
    "New experiences are great for brain plasticity! Lean into the novelty.",
    "Take a breath and process the news before reacting.",
    "Capture this moment in your journal - surprises are often turning points."
  ]
};

export default function DashboardPage() {
  const [loading, setLoading] = useState(true);
  const [moodHistory, setMoodHistory] = useState<any[]>([]);
  const [emotionCounts, setEmotionCounts] = useState<any[]>([]);
  const [successRate, setSuccessRate] = useState<any[]>([]);
  const [wellnessTip, setWellnessTip] = useState("");
  const [insights, setInsights] = useState<string[]>([]);
  const [streak, setStreak] = useState(0);
  const [timeOfDayInsights, setTimeOfDayInsights] = useState<any[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterEmotion, setFilterEmotion] = useState("all");
  const [challenges, setChallenges] = useState<any[]>([]);
  const router = useRouter();

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        router.push("/auth");
        return;
      }

        // Fetch mood entries with recommendations
        const { data: entries, error: entriesError } = await supabase
          .from("mood_entries")
          .select("*, recommendations(*)")
          .eq("user_id", user.id)
          .order("created_at", { ascending: true });

      if (entriesError) throw entriesError;
      
      const processedEntries = entries.map(e => ({
        ...e,
        date: new Date(e.created_at).toLocaleDateString(),
        confidence: Math.round(e.confidence * 100)
      }));
      setMoodHistory(processedEntries);

      // Fetch Challenges
      const { data: userChallenges, error: challengesError } = await supabase
        .from("wellness_challenges")
        .select("*")
        .eq("user_id", user.id);
      
      if (challengesError) throw challengesError;

      if (!userChallenges || userChallenges.length === 0) {
        // Seed default challenges with upsert to prevent duplicates
        const defaultChallenges = [
          { user_id: user.id, title: "The Consistency Kickstart", description: "Log your mood for 3 days in a row.", target_count: 3, category: "consistency" },
          { user_id: user.id, title: "Mindful Moments", description: "Complete 5 mood captures.", target_count: 5, category: "mindfulness" },
          { user_id: user.id, title: "Positivity Quest", description: "Achieve a 'Happy' emotion 3 times.", target_count: 3, category: "mood" }
        ];
        
        const { data: seeded, error: seedError } = await supabase
          .from("wellness_challenges")
          .upsert(defaultChallenges, { onConflict: 'user_id, title' })
          .select();
          
        if (seedError) {
          console.error("Seeding error:", seedError);
          // If upsert failed, try fetching again
          const { data: retryFetch } = await supabase
            .from("wellness_challenges")
            .select("*")
            .eq("user_id", user.id);
          setChallenges(retryFetch || []);
        } else {
          setChallenges(seeded || []);
        }
      } else {
        setChallenges(userChallenges);
      }

      // 1. Calculate Streak
      const uniqueDates = Array.from(new Set(entries.map(e => 
        new Date(e.created_at).toDateString()
      ))).map(d => new Date(d)).sort((a, b) => b.getTime() - a.getTime());

      let currentStreak = 0;
      if (uniqueDates.length > 0) {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);

        const lastEntryDate = new Date(uniqueDates[0]);
        lastEntryDate.setHours(0, 0, 0, 0);

        if (lastEntryDate.getTime() === today.getTime() || lastEntryDate.getTime() === yesterday.getTime()) {
          currentStreak = 1;
          for (let i = 0; i < uniqueDates.length - 1; i++) {
            const current = new Date(uniqueDates[i]);
            current.setHours(0, 0, 0, 0);
            const next = new Date(uniqueDates[i + 1]);
            next.setHours(0, 0, 0, 0);
            
            const diffDays = Math.round((current.getTime() - next.getTime()) / (1000 * 60 * 60 * 24));
            if (diffDays === 1) {
              currentStreak++;
            } else {
              break;
            }
          }
        }
      }
      setStreak(currentStreak);

      // 2. Calculate Time-of-Day Insights
      const timeGroups: Record<string, Record<string, number>> = {
        "Morning (5-11)": {},
        "Afternoon (12-16)": {},
        "Evening (17-21)": {},
        "Night (22-4)": {}
      };

      entries.forEach(e => {
        const hour = new Date(e.created_at).getHours();
        let period = "";
        if (hour >= 5 && hour < 12) period = "Morning (5-11)";
        else if (hour >= 12 && hour < 17) period = "Afternoon (12-16)";
        else if (hour >= 17 && hour < 22) period = "Evening (17-21)";
        else period = "Night (22-4)";
        
        timeGroups[period][e.emotion] = (timeGroups[period][e.emotion] || 0) + 1;
      });

      const timeInsights = Object.entries(timeGroups).map(([period, emotions]) => {
        const topEmotion = Object.entries(emotions).sort((a, b) => b[1] - a[1])[0];
        return {
          period,
          emotion: topEmotion ? topEmotion[0] : "N/A",
          count: topEmotion ? topEmotion[1] : 0,
          icon: period.includes("Morning") ? Sunrise : period.includes("Afternoon") ? Sun : period.includes("Evening") ? Sunset : Moon
        };
      });
      setTimeOfDayInsights(timeInsights);

      // 3. Calculate emotion frequency
      const counts: Record<string, number> = {};
      entries.forEach(e => {
        counts[e.emotion] = (counts[e.emotion] || 0) + 1;
      });
      setEmotionCounts(Object.entries(counts).map(([name, value]) => ({ name, value })));

      // 4. Fetch recommendations feedback
      const { data: recs, error: recsError } = await supabase
        .from("recommendations")
        .select("feedback_score, category")
        .not("feedback_score", "eq", 0);

      if (recsError) throw recsError;
      
      const success: Record<string, { total: number, positive: number }> = {};
      recs.forEach(r => {
        if (!success[r.category]) success[r.category] = { total: 0, positive: 0 };
        success[r.category].total += 1;
        if (r.feedback_score === 1) success[r.category].positive += 1;
      });

        setSuccessRate(Object.entries(success).map(([name, stats]) => ({
          name,
          rate: Math.round((stats.positive / stats.total) * 100)
        })));

        // 5. Generate Insights & Tips
        if (entries.length > 0) {
          const lastEntry = entries[entries.length - 1];
          const tips = WELLNESS_TIPS[lastEntry.emotion as keyof typeof WELLNESS_TIPS] || WELLNESS_TIPS["neutral"];
          setWellnessTip(tips[Math.floor(Math.random() * tips.length)]);

          const newInsights = [];
          const topEmotion = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
          newInsights.push(`You've been feeling ${topEmotion[0]} most frequently (${topEmotion[1]} times).`);
          
          if (entries.length >= 3) {
            const recentEmotions = entries.slice(-3).map(e => e.emotion);
            if (new Set(recentEmotions).size === 1) {
              newInsights.push(`You've been in a consistent ${recentEmotions[0]} state lately.`);
            }
          }
          if (currentStreak >= 3) {
            newInsights.push(`Amazing! You have a ${currentStreak}-day consistency streak.`);
          }
          setInsights(newInsights);
        }

      } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const filteredHistory = moodHistory.filter(entry => {
    const matchesSearch = entry.note?.toLowerCase().includes(searchTerm.toLowerCase()) || 
                         entry.emotion.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesEmotion = filterEmotion === "all" || entry.emotion === filterEmotion;
    return matchesSearch && matchesEmotion;
  });

  if (loading) {
    return (
      <div className="flex h-[70vh] items-center justify-center">
        <Loader2 className="w-10 h-10 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-8 pb-12">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Your Wellness Dashboard</h1>
          <p className="text-muted-foreground">Tracking your emotional journey over time.</p>
        </div>
        <div className="flex gap-4">
          <Card className="px-4 py-2 flex items-center gap-2 border-primary/20 bg-primary/5">
            <TrendingUp className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium">{moodHistory.length} Sessions</span>
          </Card>
          <Card className="px-4 py-2 flex items-center gap-2 border-orange-500/20 bg-orange-500/5">
            <Flame className="w-4 h-4 text-orange-500" />
            <span className="text-sm font-medium">{streak} Day Streak</span>
          </Card>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        <Card className="md:col-span-2 border-primary/20 bg-primary/5">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-primary" />
              <CardTitle className="text-lg">Daily Wellness Tip</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-xl font-medium text-primary-foreground bg-primary/20 p-4 rounded-xl border border-primary/30">
              "{wellnessTip}"
            </p>
          </CardContent>
        </Card>

        <Card className="md:col-span-3">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Trophy className="w-5 h-5 text-amber-500" />
                <CardTitle className="text-lg">Wellness Challenges</CardTitle>
              </div>
              <Badge variant="outline" className="border-amber-500/30 text-amber-600">
                {challenges.filter(c => c.status === 'completed').length} / {challenges.length} Completed
              </Badge>
            </div>
            <CardDescription>Small steps towards a healthier emotional state.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid sm:grid-cols-3 gap-4">
              {challenges.map((challenge) => (
                <div key={challenge.id} className="p-4 rounded-xl border bg-background/50 space-y-3 relative overflow-hidden group">
                  <div className="flex items-start justify-between relative z-10">
                    <div className="p-2 rounded-lg bg-primary/10">
                      {challenge.category === 'mood' ? <Heart className="w-4 h-4 text-primary" /> : 
                       challenge.category === 'consistency' ? <Target className="w-4 h-4 text-primary" /> : 
                       <Brain className="w-4 h-4 text-primary" />}
                    </div>
                    {challenge.status === 'completed' && <CheckCircle className="w-4 h-4 text-green-500" />}
                  </div>
                  <div className="space-y-1 relative z-10">
                    <h4 className="text-sm font-bold leading-tight">{challenge.title}</h4>
                    <p className="text-[10px] text-muted-foreground leading-relaxed line-clamp-2">{challenge.description}</p>
                  </div>
                  <div className="space-y-2 relative z-10">
                    <div className="flex justify-between text-[10px] font-medium">
                      <span>Progress</span>
                      <span>{challenge.current_count} / {challenge.target_count}</span>
                    </div>
                    <Progress value={(challenge.current_count / challenge.target_count) * 100} className="h-1.5" />
                  </div>
                  {challenge.status === 'completed' && (
                    <div className="absolute inset-0 bg-green-500/5 backdrop-blur-[1px] pointer-events-none" />
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-primary" />
              <CardTitle className="text-lg">AI Insights</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {insights.map((insight, i) => (
              <div key={i} className="flex gap-2 items-start text-sm">
                <div className="mt-1 p-1 rounded-full bg-primary/10 text-primary">
                  <TrendingUp className="w-3 h-3" />
                </div>
                <p>{insight}</p>
              </div>
            ))}
            {insights.length === 0 && <p className="text-sm text-muted-foreground">Keep tracking to unlock deeper insights.</p>}
          </CardContent>
        </Card>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border-indigo-500/20">
          <CardHeader className="p-4 pb-2">
            <CardDescription className="text-indigo-600 dark:text-indigo-400 font-bold uppercase text-[10px] tracking-wider">Top Mood</CardDescription>
            <CardTitle className="text-2xl capitalize">
              {emotionCounts.sort((a, b) => b.value - a.value)[0]?.name || "None"}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card className="bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border-emerald-500/20">
          <CardHeader className="p-4 pb-2">
            <CardDescription className="text-emerald-600 dark:text-emerald-400 font-bold uppercase text-[10px] tracking-wider">Helpful Rate</CardDescription>
            <CardTitle className="text-2xl">
              {successRate.length > 0 ? `${Math.round(successRate.reduce((acc, curr) => acc + curr.rate, 0) / successRate.length)}%` : "0%"}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card className="bg-gradient-to-br from-amber-500/10 to-orange-500/10 border-amber-500/20">
          <CardHeader className="p-4 pb-2">
            <CardDescription className="text-amber-600 dark:text-amber-400 font-bold uppercase text-[10px] tracking-wider">Streak</CardDescription>
            <CardTitle className="text-2xl">{streak} Days</CardTitle>
          </CardHeader>
        </Card>
        <Card className="bg-gradient-to-br from-rose-500/10 to-pink-500/10 border-rose-500/20">
          <CardHeader className="p-4 pb-2">
            <CardDescription className="text-rose-600 dark:text-rose-400 font-bold uppercase text-[10px] tracking-wider">Current State</CardDescription>
            <CardTitle className="text-2xl">
              {moodHistory[moodHistory.length - 1]?.mood || "Unknown"}
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-primary" />
              <CardTitle>Mood Confidence Over Time</CardTitle>
            </div>
            <CardDescription>Visualizing how clear your emotional expressions have been.</CardDescription>
          </CardHeader>
          <CardContent className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={moodHistory}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" />
                <YAxis unit="%" />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#1f2937", border: "none", borderRadius: "8px" }}
                  itemStyle={{ color: "#fff" }}
                />
                <Line type="monotone" dataKey="confidence" stroke="#8884d8" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-primary" />
              <CardTitle>Emotion Distribution</CardTitle>
            </div>
            <CardDescription>Frequency of detected emotions.</CardDescription>
          </CardHeader>
          <CardContent className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={emotionCounts}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {emotionCounts.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Clock className="w-5 h-5 text-primary" />
                <CardTitle>Time-of-Day Insights</CardTitle>
              </div>
              <CardDescription>When do you feel certain emotions?</CardDescription>
            </CardHeader>
            <CardContent className="h-[300px]">
              <ScrollArea className="h-full pr-4">
                <div className="space-y-4">
                  {timeOfDayInsights.map((insight, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-muted/30 border">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-full bg-primary/10">
                          <insight.icon className="w-4 h-4 text-primary" />
                        </div>
                        <div>
                          <p className="text-sm font-semibold">{insight.period}</p>
                          <p className="text-[10px] text-muted-foreground">Most common emotion</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant="secondary" className="capitalize">{insight.emotion}</Badge>
                        <p className="text-[10px] text-muted-foreground mt-1">{insight.count} entries</p>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          <Card className="lg:col-span-3">

          <CardHeader>
            <div className="flex items-center gap-2">
              <Heart className="w-5 h-5 text-primary" />
              <CardTitle>Recommendation Success Rate</CardTitle>
            </div>
            <CardDescription>Which categories are helping you the most?</CardDescription>
          </CardHeader>
          <CardContent className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={successRate}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" />
                <YAxis unit="%" />
                <Tooltip />
                <Bar dataKey="rate" fill="#82ca9d" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

        <Card>
          <CardHeader>
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-primary" />
                <CardTitle>Mood Diary</CardTitle>
              </div>
              <div className="flex flex-col sm:flex-row gap-2">
                <div className="relative">
                  <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search notes or emotions..."
                    className="pl-8 w-full sm:w-[250px] h-9 text-xs"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                </div>
                <Select value={filterEmotion} onValueChange={setFilterEmotion}>
                  <SelectTrigger className="w-full sm:w-[130px] h-9 text-xs">
                    <div className="flex items-center gap-2">
                      <Filter className="w-3 h-3" />
                      <SelectValue placeholder="Emotion" />
                    </div>
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Emotions</SelectItem>
                    {["happy", "sad", "neutral", "angry", "surprise"].map(emo => (
                      <SelectItem key={emo} value={emo} className="capitalize">{emo}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <CardDescription>Your history of emotions and journal notes.</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[400px] pr-4">
              <div className="space-y-4">
                {[...filteredHistory].reverse().map((entry, i) => (

                  <div key={i} className="p-4 rounded-xl border bg-muted/30 space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="capitalize">{entry.emotion}</Badge>
                          {entry.sentiment_label && (
                            <Badge variant="secondary" className="capitalize bg-primary/10 text-primary border-primary/20 text-[9px] h-5">
                              <Sparkles className="w-2.5 h-2.5 mr-1" />
                              {entry.sentiment_label} Tone
                            </Badge>
                          )}
                          <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {entry.date}
                          </span>
                        </div>
                        <Badge variant="secondary" className="text-[10px]">{entry.confidence}% Conf.</Badge>
                      </div>
                    
                    {entry.note ? (
                      <p className="text-sm italic text-foreground/80 bg-background/50 p-2 rounded-md border-l-2 border-primary">
                        "{entry.note}"
                      </p>
                    ) : (
                      <p className="text-xs text-muted-foreground italic">No journal note added.</p>
                    )}

                    {entry.recommendations && entry.recommendations.length > 0 && (
                      <div className="pt-2 border-t border-border/50 space-y-2">
                        <p className="text-[10px] font-bold uppercase text-muted-foreground flex items-center gap-1">
                          <Sparkles className="w-3 h-3" />
                          Recommendations
                        </p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                          {entry.recommendations.map((rec: any, idx: number) => (
                            <div key={idx} className="flex flex-col gap-1 p-2 rounded-lg bg-background/40 border border-border/30">
                              <div className="flex items-center justify-between">
                                <span className="text-[9px] font-bold uppercase text-primary/70">{rec.category}</span>
                                {rec.link && (
                                  <Button 
                                    variant="link" 
                                    size="sm" 
                                    className="p-0 h-auto text-primary gap-0.5 font-bold text-[9px] hover:underline"
                                    asChild
                                  >
                                    <a 
                                      href={rec.link} 
                                      target="_blank" 
                                      rel="noopener noreferrer"
                                    >
                                      {rec.category === "music" ? "Listen" : "Watch"}
                                      <ExternalLink className="w-2.5 h-2.5" />
                                    </a>
                                  </Button>
                                )}
                              </div>
                              <p className="text-[11px] font-semibold leading-tight line-clamp-1">{rec.content}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
              ))}
              {moodHistory.length === 0 && (
                <div className="text-center py-8">
                  <MessageSquare className="w-8 h-8 mx-auto text-muted-foreground mb-2" />
                  <p className="text-sm text-muted-foreground">No entries found yet.</p>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}
