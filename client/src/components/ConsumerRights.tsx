import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Shield, FileText, Phone, ExternalLink } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { BackButton } from "@/components/BackButton";

interface ComplaintCategory {
  name: string;
  description: string;
  law: string;
  examples: string[];
}

const complaintCategories: ComplaintCategory[] = [
  {
    name: "Food Quality & Safety",
    description: "Contaminated, spoiled, or unsafe food products",
    law: "Food Safety & Standards Act, 2006",
    examples: ["Expired products", "Foreign objects in food", "Food poisoning", "Mislabeling"]
  },
  {
    name: "Quantity & Weight Issues", 
    description: "Incorrect weights, measures, or quantities",
    law: "Legal Metrology Act, 2009",
    examples: ["Underweight packages", "Incorrect MRP", "False quantity claims", "Defective weighing scales"]
  },
  {
    name: "Misleading Advertisements",
    description: "False or deceptive marketing claims",
    law: "Consumer Protection Act, 2019",
    examples: ["False health claims", "Fake testimonials", "Hidden ingredients", "Exaggerated benefits"]
  }
];

const helplineNumbers = [
  { name: "National Consumer Helpline", number: "1915", description: "General consumer complaints" },
  { name: "FSSAI Helpline", number: "1800-11-4000", description: "Food safety issues" },
  { name: "Legal Metrology", number: "1800-11-4001", description: "Weight & measure violations" }
];

const ConsumerRights = () => {
  const [selectedCategory, setSelectedCategory] = useState<ComplaintCategory | null>(null);
  const [complaintText, setComplaintText] = useState("");
  const [productName, setProductName] = useState("");
  const { toast } = useToast();

  const generateComplaintTemplate = () => {
    if (!selectedCategory || !productName) {
      toast({
        title: "Missing Information",
        description: "Please select a category and enter product name",
        variant: "destructive"
      });
      return;
    }

    const template = `
Subject: Complaint regarding ${productName} - ${selectedCategory.name}

Dear Sir/Madam,

I am writing to file a formal complaint under the ${selectedCategory.law} regarding the following issue:

Product Name: ${productName}
Issue Category: ${selectedCategory.name}
Date of Purchase: [Enter Date]
Place of Purchase: [Enter Store Name/Location]

Complaint Details:
${complaintText || '[Describe your specific issue in detail]'}

I request immediate action as per my rights under the Consumer Protection Act, 2019.

Attached Documents:
- Purchase receipt
- Product photos
- Medical certificate (if applicable)

I look forward to your prompt response and resolution.

Yours sincerely,
[Your Name]
[Your Contact Details]
    `;

    navigator.clipboard.writeText(template.trim());
    toast({
      title: "Template Copied!",
      description: "Complaint template copied to clipboard. Paste it in your complaint form."
    });
  };

  return (
    <div className="min-h-screen bg-gradient-hero">
      <div className="container mx-auto px-6 py-20">
        <BackButton />
        
        <div className="text-center mb-12 text-white">
          <Shield className="h-16 w-16 mx-auto mb-6" />
          <h1 className="text-5xl font-bold mb-4">Consumer Rights & Legal Help</h1>
          <p className="text-xl text-white/90 max-w-2xl mx-auto">
            Know your rights, file complaints, and fight food fraud with our legal guidance
          </p>
        </div>

        <div className="max-w-6xl mx-auto space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {complaintCategories.map((category, index) => (
              <Card 
                key={index}
                className={`bg-white/95 backdrop-blur-sm cursor-pointer transition-all ${
                  selectedCategory?.name === category.name ? 'ring-2 ring-primary shadow-medium' : 'hover:shadow-medium'
                }`}
                onClick={() => setSelectedCategory(category)}
              >
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5 text-orange-600" />
                    {category.name}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">{category.description}</p>
                  <Badge variant="outline" className="mb-3">{category.law}</Badge>
                  <div>
                    <h4 className="font-semibold text-sm mb-2">Examples:</h4>
                    <ul className="text-xs text-muted-foreground space-y-1">
                      {category.examples.map((example, i) => (
                        <li key={i}>• {example}</li>
                      ))}
                    </ul>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {selectedCategory && (
            <Card className="bg-white/95 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Generate Complaint Template - {selectedCategory.name}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Product Name</label>
                    <Input
                      placeholder="Enter product name"
                      value={productName}
                      onChange={(e) => setProductName(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-2 block">Applicable Law</label>
                    <Input value={selectedCategory.law} disabled />
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Describe Your Issue</label>
                  <Textarea
                    placeholder="Provide detailed description of the problem..."
                    value={complaintText}
                    onChange={(e) => setComplaintText(e.target.value)}
                    rows={4}
                  />
                </div>

                <Button 
                  onClick={generateComplaintTemplate}
                  className="w-full bg-gradient-card text-white"
                  size="lg"
                >
                  Generate & Copy Complaint Template
                </Button>
              </CardContent>
            </Card>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="bg-white/95 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Phone className="h-5 w-5" />
                  Emergency Helplines
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {helplineNumbers.map((helpline, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <div className="font-semibold">{helpline.name}</div>
                        <div className="text-sm text-muted-foreground">{helpline.description}</div>
                      </div>
                      <Badge className="bg-green-100 text-green-800 font-mono">
                        {helpline.number}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/95 backdrop-blur-sm">
              <CardHeader>
                <CardTitle>Know Your Rights</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4 text-sm">
                  <div className="p-3 border-l-4 border-primary bg-primary/5">
                    <strong>Right to Safety:</strong> Protection against hazardous goods and services
                  </div>
                  <div className="p-3 border-l-4 border-primary bg-primary/5">
                    <strong>Right to Information:</strong> Complete product information including ingredients, price, quality
                  </div>
                  <div className="p-3 border-l-4 border-primary bg-primary/5">
                    <strong>Right to Choice:</strong> Access to variety of goods at competitive prices
                  </div>
                  <div className="p-3 border-l-4 border-primary bg-primary/5">
                    <strong>Right to be Heard:</strong> Voice complaints and be assured of fair treatment
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-white/95 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Quick Links & Resources</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-2">
                  <ExternalLink className="h-5 w-5" />
                  <span className="text-sm">National Consumer Portal</span>
                </Button>
                <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-2">
                  <ExternalLink className="h-5 w-5" />
                  <span className="text-sm">FSSAI Complaint Portal</span>
                </Button>
                <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-2">
                  <ExternalLink className="h-5 w-5" />
                  <span className="text-sm">Consumer Court Finder</span>
                </Button>
                <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-2">
                  <ExternalLink className="h-5 w-5" />
                  <span className="text-sm">Legal Aid Services</span>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ConsumerRights;